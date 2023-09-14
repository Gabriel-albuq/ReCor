# consumers.py

import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import async_to_sync

class SeuConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    @async_to_sync
    async def update_contador(self, event):
        contador = event['contador']
        await self.send(text_data=json.dumps({'contador': contador}))
