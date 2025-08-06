from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from trading_buddy.nlp.parser import NLPParser
from trading_buddy.schemas.hypothesis import Hypothesis

router = APIRouter()


class ParseRequest(BaseModel):
    text: str


@router.post("/parse", response_model=Hypothesis)
async def parse_text(request: ParseRequest):
    """
    Parse natural language text into structured hypothesis.
    
    Example:
    "Watching the MACD on SPY — about to cross on the 5 after a double tap following a big drop. thoughts?"
    """
    try:
        parser = NLPParser()
        hypothesis = parser.parse(request.text)
        return hypothesis
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse text: {str(e)}")