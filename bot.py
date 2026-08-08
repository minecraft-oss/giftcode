import os
import io
import asyncio
import logging
import re
from datetime import datetime
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
from PIL import Image
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# Load environment variables
load_dotenv()

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
PHONE_NUMBER = os.getenv("PHONE_NUMBER")

# Initialize RapidOCR Engine globally (avoid re-loading ONNX model per request)
ocr_engine = RapidOCR()

def parse_single_id(chat_id_str):
    """Parses a single chat ID or username into int or string."""
    if not chat_id_str:
        return None
    chat_id_str = chat_id_str.strip()
    try:
        return int(chat_id_str)
    except ValueError:
        return chat_id_str

def parse_chat_ids(chat_ids_str):
    """Parses a comma-separated string of IDs/usernames."""
    if not chat_ids_str:
        return []
    parsed_ids = []
    for cid in chat_ids_str.split(','):
        parsed = parse_single_id(cid)
        if parsed is not None:
            parsed_ids.append(parsed)
    return parsed_ids

# Parse source and target IDs correctly
SOURCE_GROUPS = parse_chat_ids(os.getenv("SOURCE_GROUP_IDS"))
TARGET_GROUP = parse_single_id(os.getenv("TARGET_GROUP_ID"))

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Telegram Client
client = TelegramClient("ocr_userbot_session", API_ID, API_HASH)

def process_image_sync(image_bytes: bytes) -> str:
    """Synchronous function to process image and extract text in RAM."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Optimize: Resize if width > 1200px
        if image.width > 1200:
            ratio = 1200.0 / image.width
            new_height = int(image.height * ratio)
            image = image.resize((1200, new_height), Image.Resampling.LANCZOS)
            
        img_np = np.array(image)
        
        # Run RapidOCR
        result, _ = ocr_engine(img_np)
        
        if not result:
            return ""
            
        # RapidOCR result format: [[box, text, score], ...]
        extracted_text = "\n".join([line[1] for line in result])
        return extracted_text.strip()
    
    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        return ""

@client.on(events.NewMessage(chats=SOURCE_GROUPS))
async def handle_new_image(event):
    if not event.message.photo:
        return

    # Get source channel/group name
    chat = await event.get_chat()
    chat_name = getattr(chat, 'title', getattr(chat, 'username', f"ID: {chat.id}"))

    logger.info(f"New image detected in '{chat_name}' (Msg ID: {event.message.id}). Processing...")

    try:
        # 1. Download image into RAM
        image_bytes = await client.download_media(event.message.photo, file=bytes)
        
        if not image_bytes:
            logger.warning("Failed to download image bytes.")
            return

        # 2. Run OCR in thread
        extracted_text = await asyncio.to_thread(process_image_sync, image_bytes)
        
        # 3. Filter Codes and Format Output
        now = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
        codes = re.findall(r'\b[a-zA-Z0-9]{9,12}\b', extracted_text)
        
        if not codes:
            output_msg = f"Không tìm thấy giftcode nào trong ảnh từ **{chat_name}**."
        else:
            formatted_codes = "\n".join([f"`{code}`" for code in codes])
            
            output_msg = (
                f"🕒 Time: {now}\n"
                f"═══════════════════════\n"
                f"{formatted_codes}\n"
                f"═══════════════════════\n"
                f"📍 Source: **{chat_name}**"
            )   
            
        # 4. Send to Target Group
        await client.send_message(TARGET_GROUP, output_msg)
        logger.info(f"Successfully processed and forwarded OCR result from {chat_name}.")
        
        # 5. Clean RAM
        del image_bytes

    except FloodWaitError as e:
        logger.warning(f"FloodWaitError: Must wait {e.seconds} seconds.")
        await asyncio.sleep(e.seconds)
    except Exception as e:
        logger.error(f"Unexpected error handling message: {e}")

async def main():
    logger.info("Starting userbot...")
    await client.start(phone=PHONE_NUMBER)
    
    # Warm up cache so Telethon recognizes all group/channel IDs instantly
    logger.info("Fetching dialogs to load entity cache...")
    await client.get_dialogs()
    
    logger.info(f"Userbot is running and listening to {len(SOURCE_GROUPS)} source(s).")
    await client.run_until_disconnected()

if __name__ == '__main__':
    asyncio.run(main())