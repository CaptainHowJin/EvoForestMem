from __future__ import annotations
from datetime import datetime, timezone
from evoforestmem.log import tags as tagmod
from evoforestmem.log.registry import Registry
from evoforestmem.log.store import MessageLog

def _to_ts(time_str: str) -> float:
    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()

def build_log(messages: list[dict]) -> tuple[MessageLog, Registry, Registry, dict[str, str]]:

    channels = Registry(kind='channel')
    speakers = Registry(kind='speaker')
    for row in messages:
        channels.add(str(row['group']))
        speakers.add(str(row['speaker']))
    known = speakers.names
    log = MessageLog()
    id_map: dict[str, str] = {}
    for row in messages:
        text = str(row.get('dialogue') or row.get('text') or '').strip()
        stored = log.append(text=text, speaker_key=speakers.key(str(row['speaker'])), channel_key=channels.key(str(row['group'])), ts=_to_ts(str(row['time'])), reply_to=None, tags=tagmod.derive(text, known))
        id_map[str(row['msg_id'])] = stored.msg_id
    return (log, channels, speakers, id_map)
