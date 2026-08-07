from __future__ import annotations

import ftplib
import io
import logging
import posixpath
import re
import uuid
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Self

from .config import Settings
from .models import RemoteFile, RemotePair

logger = logging.getLogger(__name__)

_UNIX_LIST_LINE = re.compile(
    r"^(?P<kind>[bcdlps-])(?P<permissions>[^\s]{9})[.+@]?\s+"
    r"(?P<links>\d+)\s+(?P<owner>\S+)\s+(?P<group>\S+)\s+"
    r"(?P<size>\d+)\s+(?P<month>[A-Za-z]{3})\s+"
    r"(?P<day>\d{1,2})\s+(?P<when>\d{2}:\d{2}|\d{4})\s+"
    r"(?P<name>.+)$"
)


def _remote_join(directory: str, name: str) -> str:
    if name.startswith("/"):
        return posixpath.normpath(name)
    if directory in ("", "."):
        return posixpath.normpath(name)
    return posixpath.normpath(posixpath.join(directory, name))


def build_pairs(
    files: Iterable[RemoteFile], image_extensions: tuple[str, ...]
) -> list[RemotePair]:
    images: dict[tuple[str, str], list[RemoteFile]] = defaultdict(list)
    xml_files: dict[tuple[str, str], list[RemoteFile]] = defaultdict(list)
    allowed = {extension.casefold() for extension in image_extensions}

    for remote in files:
        path = PurePosixPath(remote.path)
        key = (str(path.parent), path.stem.casefold())
        suffix = path.suffix.casefold()
        if suffix in allowed:
            images[key].append(remote)
        elif suffix == ".xml":
            xml_files[key].append(remote)

    pairs: list[RemotePair] = []
    for key in sorted(images):
        image_matches = images[key]
        xml_matches = xml_files.get(key, [])
        if len(image_matches) == 1 and len(xml_matches) == 1:
            pairs.append(RemotePair(image=image_matches[0], xml=xml_matches[0]))
            continue
        if xml_matches:
            logger.warning(
                "Skipping ambiguous FTP pair directory=%s stem=%s images=%d xml=%d",
                key[0],
                key[1],
                len(image_matches),
                len(xml_matches),
            )
    return pairs


def parse_unix_list_line(line: str) -> tuple[str, dict[str, str]] | None:
    match = _UNIX_LIST_LINE.match(line)
    if match is None:
        return None
    kind = match.group("kind")
    entry_type = {"-": "file", "d": "dir", "l": "link"}.get(kind, "other")
    return (
        match.group("name"),
        {
            "type": entry_type,
            "size": match.group("size"),
            "modify": (
                f"LIST:{match.group('month')}:{match.group('day')}:"
                f"{match.group('when')}"
            ),
        },
    )


class FtpClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ftp: ftplib.FTP | None = None

    def __enter__(self) -> Self:
        ftp = ftplib.FTP(timeout=self.settings.ftp_timeout_seconds)
        ftp.encoding = self.settings.ftp_encoding
        ftp.connect(self.settings.ftp_host, self.settings.ftp_port)
        ftp.login(self.settings.ftp_user, self.settings.ftp_password)
        ftp.set_pasv(self.settings.ftp_passive)
        self.ftp = ftp
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.ftp is None:
            return
        try:
            self.ftp.quit()
        except (OSError, EOFError, ftplib.Error):
            try:
                self.ftp.close()
            except OSError:
                pass
        finally:
            self.ftp = None

    def list_files(self, root: str, recursive: bool) -> list[RemoteFile]:
        ftp = self._connected()
        pending = [posixpath.normpath(root or "/")]
        visited: set[str] = set()
        result: list[RemoteFile] = []

        while pending:
            directory = pending.pop()
            if directory in visited:
                continue
            visited.add(directory)
            entries = self._list_directory(ftp, directory)
            for name, facts in entries:
                if name in {".", ".."}:
                    continue
                remote_path = _remote_join(directory, name)
                entry_type = facts.get("type", "").casefold()
                if entry_type in {"cdir", "pdir"}:
                    continue
                if entry_type == "dir":
                    if recursive:
                        pending.append(remote_path)
                    continue
                if entry_type not in {"file", ""}:
                    continue
                size = self._optional_size(facts.get("size"))
                if size is None:
                    try:
                        size = ftp.size(remote_path)
                    except ftplib.Error:
                        size = None
                modified = facts.get("modify")
                if not modified:
                    try:
                        response = ftp.sendcmd(f"MDTM {remote_path}")
                        modified = response.removeprefix("213 ").strip()
                    except ftplib.Error:
                        modified = None
                result.append(
                    RemoteFile(path=remote_path, size=size, modified=modified)
                )
        return result

    def _list_directory(
        self, ftp: ftplib.FTP, directory: str
    ) -> list[tuple[str, dict[str, str]]]:
        try:
            return list(ftp.mlsd(directory, facts=["type", "size", "modify"]))
        except (ftplib.error_perm, ftplib.error_temp, AttributeError) as exc:
            logger.debug("MLSD failed for %s, trying LIST: %s", directory, exc)
        try:
            return self._list_directory_list(ftp, directory)
        except (ftplib.Error, ValueError) as exc:
            logger.debug("LIST failed for %s, using NLST: %s", directory, exc)
            return self._list_directory_nlst(ftp, directory)

    @staticmethod
    def _list_directory_list(
        ftp: ftplib.FTP, directory: str
    ) -> list[tuple[str, dict[str, str]]]:
        lines: list[str] = []
        ftp.retrlines(f"LIST {directory}", lines.append)
        entries: list[tuple[str, dict[str, str]]] = []
        for line in lines:
            if line.casefold().startswith("total "):
                continue
            parsed = parse_unix_list_line(line)
            if parsed is None:
                raise ValueError(f"Unsupported FTP LIST line: {line[:200]}")
            entries.append(parsed)
        return entries

    def _list_directory_nlst(
        self, ftp: ftplib.FTP, directory: str
    ) -> list[tuple[str, dict[str, str]]]:
        current = ftp.pwd()
        entries: list[tuple[str, dict[str, str]]] = []
        try:
            ftp.cwd(directory)
            for raw_name in ftp.nlst():
                name = posixpath.basename(raw_name.rstrip("/"))
                if name in {"", ".", ".."}:
                    continue
                facts: dict[str, str] = {}
                try:
                    ftp.cwd(name)
                except ftplib.Error:
                    facts["type"] = "file"
                    try:
                        size = ftp.size(name)
                        if size is not None:
                            facts["size"] = str(size)
                    except ftplib.Error:
                        pass
                    try:
                        response = ftp.sendcmd(f"MDTM {name}")
                        facts["modify"] = response.removeprefix("213 ").strip()
                    except ftplib.Error:
                        pass
                else:
                    facts["type"] = "dir"
                    ftp.cwd("..")
                entries.append((name, facts))
        finally:
            ftp.cwd(current)
        return entries

    def download_to(self, remote_path: str, local_path: Path) -> None:
        ftp = self._connected()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_path.with_name(f".{local_path.name}.{uuid.uuid4().hex}.part")
        try:
            with temporary.open("wb") as handle:
                ftp.retrbinary(f"RETR {remote_path}", handle.write)
            temporary.replace(local_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def download_bytes(self, remote_path: str) -> bytes:
        ftp = self._connected()
        buffer = io.BytesIO()
        ftp.retrbinary(f"RETR {remote_path}", buffer.write)
        return buffer.getvalue()

    def upload_atomic(self, remote_path: str, content: bytes) -> None:
        ftp = self._connected()
        token = uuid.uuid4().hex
        temporary = f"{remote_path}.tmp-{token}"
        backup = f"{remote_path}.bak-{token}"
        ftp.storbinary(f"STOR {temporary}", io.BytesIO(content))
        try:
            ftp.rename(temporary, remote_path)
            return
        except ftplib.Error as initial_error:
            try:
                ftp.rename(remote_path, backup)
            except ftplib.Error:
                self._safe_delete(temporary)
                raise initial_error

            try:
                ftp.rename(temporary, remote_path)
            except Exception:
                try:
                    ftp.rename(backup, remote_path)
                finally:
                    self._safe_delete(temporary)
                raise
            else:
                self._safe_delete(backup)

    def _safe_delete(self, remote_path: str) -> None:
        try:
            self._connected().delete(remote_path)
        except ftplib.Error:
            pass

    @staticmethod
    def _optional_size(value: str | None) -> int | None:
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _connected(self) -> ftplib.FTP:
        if self.ftp is None:
            raise RuntimeError("FTP client is not connected")
        return self.ftp
