"""Civil-time conversion that refuses ambiguous/nonexistent DST instants."""
from datetime import datetime,timezone
from zoneinfo import ZoneInfo,ZoneInfoNotFoundError


def local_time_info(raw,zone):
    if not raw:
        return (None,None,'missing_time')
    try:
        dt=datetime.fromisoformat(str(raw).replace('Z','+00:00'))
    except ValueError:
        return (str(raw),None,'invalid_time')
    try:
        tz=ZoneInfo(zone) if zone else None
    except (ZoneInfoNotFoundError,ValueError):
        tz=None
    if dt.tzinfo:
        local=dt.astimezone(tz) if tz else dt
        return (local.replace(tzinfo=None).isoformat(),dt.astimezone(timezone.utc).isoformat(),
                'valid' if tz else 'aware_without_zone')
    if tz is None:
        return (dt.isoformat(),None,'missing_or_invalid_zone')
    candidates=[dt.replace(tzinfo=tz,fold=f) for f in (0,1)]
    valid=[v for v in candidates if v.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None)==dt]
    if not valid:
        return (dt.isoformat(),None,'nonexistent_local_time')
    if len({v.utcoffset() for v in valid})>1:
        return (dt.isoformat(),None,'ambiguous_local_time')
    return (dt.isoformat(),valid[0].astimezone(timezone.utc).isoformat(),'valid')
