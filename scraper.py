# scraper.py (patched)
import logging
import re
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Playwright sync API (we'll use .start()/.stop() to avoid context-manager init bugs)
from playwright.sync_api import sync_playwright

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini model via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ.get('GOOGLE_API_KEY')
)

def call_gemini_model(prompt, content_text):
    """
    Calls Gemini via LangChain wrapper to generate a podcast-style summary.
    """
    try:
        input_text = f"""
        {prompt}

        Raw Scraped Content:
        {content_text}
        """
        response = llm.invoke(input_text)
        return response.content.strip()
    except Exception as e:
        logger.exception("Error calling Gemini model")
        return ""

# ---------------- SCRAPER ---------------- #
def scrape_new_posts():
    """
    Scrapes new posts from The Rundown AI and saves raw content in Markdown + JSON.
    Uses sync_playwright().start() / .stop() to avoid PlaywrightContextManager issues.
    """
    base_url = "https://www.therundown.ai"
    last_url_file = "last_scraped_url.txt"
    output_file = "news_content.md"
    json_file = "news_content.json"

    # Read last scraped URL if exists
    last_scraped_url = None
    if os.path.exists(last_url_file):
        with open(last_url_file, 'r', encoding='utf-8') as f:
            last_scraped_url = f.read().strip()

    playwright = None
    browser = None
    try:
        # start Playwright explicitly (fallback to avoid the context-manager bug)
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        logger.info("Navigating to base URL...")
        page.goto(base_url, wait_until="networkidle")

        # SELECTOR: adjust if site structure changes
        post_links = page.query_selector_all('div.grid div.transparent a[data-discover="true"][href^="/p/"]')
        post_urls = []
        for link in post_links:
            relative_url = link.get_attribute('href')
            if not relative_url:
                continue
            full_url = base_url + relative_url if relative_url.startswith('/p/') else relative_url
            post_urls.append(full_url)

        if not post_urls:
            logger.info("No post links found.")
            return

        # Build list of new posts up to last_scraped_url
        new_post_urls = []
        for url in post_urls:
            if url == last_scraped_url:
                break
            new_post_urls.append(url)

        if not new_post_urls:
            logger.info("No new posts found.")
            return

        markdown_content = "# Scraped News Content\n\n"
        all_posts_content = []

        for url in new_post_urls:
            logger.info(f"Scraping new post: {url}")
            page.goto(url, wait_until="networkidle")

            headlines = page.query_selector_all('h1, h2, h3, h4, h5, h6')
            paragraphs = page.query_selector_all('p')
            list_items = page.query_selector_all('li')

            post_content = {
                'url': url,
                'headlines': [],
                'paragraphs': [],
                'list_items': []
            }

            collect_content = True

            def add_if_allowed(text, target_list):
                nonlocal collect_content
                if not text:
                    return
                # stop collecting if "that's it for today" phrase appears
                if re.search(r"that['’]s\s*it\s*for\s*today", text, re.IGNORECASE):
                    collect_content = False
                    return
                # resume/ensure collecting on "latest developments" lines
                if re.search(r"latest\s*developments", text, re.IGNORECASE):
                    collect_content = True
                if collect_content:
                    target_list.append(text)

            for h in headlines:
                add_if_allowed(h.inner_text().strip(), post_content['headlines'])

            for ptag in paragraphs:
                add_if_allowed(ptag.inner_text().strip(), post_content['paragraphs'])

            for li in list_items:
                add_if_allowed(li.inner_text().strip(), post_content['list_items'])

            all_posts_content.append(post_content)

            markdown_content += f"## Post: {url}\n\n"
            if post_content['headlines']:
                markdown_content += "### Headlines\n" + "\n".join(f"- {h}" for h in post_content['headlines']) + "\n\n"
            if post_content['paragraphs']:
                markdown_content += "### Paragraphs\n" + "\n\n".join(post_content['paragraphs']) + "\n\n"
            if post_content['list_items']:
                markdown_content += "### List Items\n" + "\n".join(f"- {i}" for i in post_content['list_items']) + "\n\n"
            markdown_content += "---\n\n"

        # Save markdown + JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_posts_content, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Raw content saved to {output_file} and {json_file}")

        # Update last scraped URL (first in current list)
        if post_urls:
            with open(last_url_file, 'w', encoding='utf-8') as f:
                f.write(post_urls[0])
            logger.info(f"Updated last scraped URL to {post_urls[0]}")

    except Exception as e:
        logger.exception("Error while scraping new posts")
        raise
    finally:
        try:
            if browser:
                browser.close()
        except Exception:
            logger.exception("Error closing browser")
        try:
            if playwright:
                playwright.stop()
        except Exception:
            logger.exception("Error stopping playwright")


# ---------------- PODCAST SUMMARY ---------------- #
def generate_podcast_summary_from_file(md_file="news_content.md", podcast_file="podcast_summary.txt"):
    """
    Reads news_content.md and generates a podcast-style summary.
    """
    if not os.path.exists(md_file):
        logger.error("Error: %s not found.", md_file)
        return ""

    with open(md_file, "r", encoding="utf-8") as f:
        content_text = f.read()

    prompt = """
    You are an AI tasked with creating a concise, engaging, and professional podcast-style summary for a daily AI news update.
    The input is raw scraped Markdown content from The Rundown AI. 

    Instructions:
    1. Extract all the AI News.
    2. Write in a conversational, natural podcast style suitable for TTS (like a podcast host).
    3. Structure: intro → main stories → closing. BUT DO NOT WRITE THE INTRO, MAIN STORIES IN THE SUMMARY.
    4. Do not mention special characters like '*'. ALSO DO NOT MAKE THE TEXT BOLD.
    5. Ignore irrelevant boilerplate like “That’s it for today”.
    """

    summary = call_gemini_model(prompt, content_text)

    with open(podcast_file, "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info(f"🎙️ Podcast summary saved to {podcast_file}")
    return summary


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    # For quick local testing only:
    # scrape_new_posts()
    generate_podcast_summary_from_file("news_content.md")
