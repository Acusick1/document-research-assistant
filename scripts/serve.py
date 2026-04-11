import uvicorn

from research_assistant.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    log_level = settings.log_level.lower()
    uvicorn.run(
        "research_assistant.server:app",
        reload=True,
        log_level=log_level,
    )
