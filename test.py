import asyncio
from deploy_agent import run_one, BrowserPool
from utils.openai_generator import OpenAIAsyncGenerator

async def main():
    # Initialize generator and browser
    generator = OpenAIAsyncGenerator(
        base_url="http://localhost:8001/v1",
        model_name="OpenResearcher/OpenResearcher-30B-A3B",
        use_native_tools=True
    )
    browser_pool = BrowserPool(search_url="http://localhost:8000", browser_backend="local")
    # browser_pool = BrowserPool(search_url=None, browser_backend="serper")


    # Run deep research
    await run_one(
        question="Determine if Neil Patrick Harris has narrated the audiobook version of 'Henry Huggins'.",
        qid="quick_start",
        generator=generator,
        browser_pool=browser_pool,
    )

    browser_pool.cleanup("quick_start")

if __name__ == "__main__":
    asyncio.run(main())