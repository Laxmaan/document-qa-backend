"""Callback handlers used in the app."""
from typing import Any, Coroutine, Dict, List, Optional
from loguru import logger
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler

from app.schemas import ChatResponse


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message=token, type="stream",seq=-1)
        logger.debug(resp)
        await self.websocket.send_json(resp.dict())


class QuestionGenCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        logger.info("QGEN LLM start")
        logger.info(f"QGEN LLM PROMPT :{prompts}")
        resp = ChatResponse(
            sender="bot", message="Synthesizing question...", type="info",seq=-4
        )
        logger.debug(resp)
        await self.websocket.send_json(resp.dict())

    async def on_text(self, text: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Coroutine[Any, Any, None]:
        logger.debug(text,run_id,)
        return await super().on_text(text, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        
        logger.debug(f" QGen on llm token :{token}")