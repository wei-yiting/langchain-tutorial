from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PlaywrightURLLoader
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

from prompts import (
    clean_text_system_prompt,
    clean_text_human_prompt,
    translate_text_system_prompt,
    translate_text_human_prompt,
)

load_dotenv()


async def main():
    # print("Loading PDF...")
    # pdf_loader = PyPDFLoader("data/ai-system-evaluation-criteria.pdf")
    # pdf_docs = pdf_loader.load()
    # print(f"Loaded {len(pdf_docs)} pages")

    web_docs_url = "https://papiers.ai/1706.03762?referrer=luma"
    web_loader = PlaywrightURLLoader(
        [web_docs_url], remove_selectors=["header", "footer", "aside"]
    )
    web_docs = await web_loader.aload()
    print(f"Loaded {len(web_docs)} pages")

    # selected_pages = pdf_docs[:]
    # pdf_text_page_contents = [page.page_content for page in selected_pages]

    web_text_page_contents = [page.page_content for page in web_docs]

    # LLM 1: Clean text extracted from
    # gemini_2_5_flash = init_chat_model(
    #     model_provider="google_genai",
    #     model="gemini-2.5-flash",
    #     temperature=0.0,
    #     max_retries=10,
    # )

    gemini_2_5_pro = init_chat_model(
        model_provider="google_genai",
        model="gemini-2.5-pro",
        temperature=0.0,
        max_retries=10,
    )
    clean_text_prompt = ChatPromptTemplate(
        [
            ("system", clean_text_system_prompt),
            ("user", clean_text_human_prompt),
        ]
    )
    clean_text_chain = clean_text_prompt | gemini_2_5_pro

    # Merge text from multiple pages and split text into paragraphs for translation
    merged_text = []
    # responses = await clean_text_chain.abatch(
    #     pdf_text_page_contents, max_concurrency=20, return_exceptions=True
    # )

    responses = await clean_text_chain.abatch(
        web_text_page_contents, max_concurrency=20, return_exceptions=True
    )

    for response in responses:
        if response.content[-1] == ".":
            cleaned_text = response.content + "\n"
        else:
            cleaned_text = response.content
        merged_text.append(cleaned_text)
    merged_text = " ".join(merged_text)

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=5000,
        chunk_overlap=0,
        length_function=len,
    )
    paragraphs = text_splitter.split_text(merged_text)
    for paragraph in paragraphs:
        print(paragraph)
        print("-" * 100)

    # LLM 2: Translate text
    translate_text_prompt = ChatPromptTemplate(
        [
            ("system", translate_text_system_prompt),
            ("user", translate_text_human_prompt),
        ]
    )
    translate_text_chain = translate_text_prompt | gemini_2_5_pro

    responses = await translate_text_chain.abatch(
        [{"cleaned_paragraph": paragraph} for paragraph in paragraphs],
        max_concurrency=20,
        return_exceptions=True,
    )
    translated_article = "\n\n".join([response.content for response in responses])

    # Save translated article to file
    with open(
        "output/v11_web_page_2.5_pro.md",
        "w",
    ) as f:
        f.write(translated_article)


if __name__ == "__main__":
    asyncio.run(main())
