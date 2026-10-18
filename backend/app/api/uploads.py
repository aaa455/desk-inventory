from fastapi import HTTPException, UploadFile

from app.core.config import settings


async def read_upload(file: UploadFile) -> bytes:
    """限制读入的文件大小，使用完毕后释放上传文件。"""
    data = bytearray()
    try:
        while chunk := await file.read(64 * 1024):
            data.extend(chunk)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "图片文件超过允许的大小。")
    finally:
        await file.close()
    return bytes(data)
