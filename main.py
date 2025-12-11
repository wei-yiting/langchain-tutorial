from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

clean_text_system_prompt = """
    The input text is extracted from a PDF. Process the text following rules below:
    - The text contains formatting errors, broken lines, unnecessary whitespace, page numbers, footnotes, endnotes, and footers. Please ignore these artifacts, reconstruct the correct sentences.
    - The output should remain same paragraph structure as the input. If part of the text is considered as a heading and a paragraph, keep the heading and the paragraph together with a line break in between. If part of the text is considered as a list, keep the list together with a line break in between.
    - The text may be written in a mixed language of English and Chinese(Traditional). Keep the original language of the text. Don't translate the text.
    - Remove the content of tables and images, yet remain the captions of tables and images.
    - For the text which are programming code blocks, delimit the code blocks with ```.
    - For the text which is part of math equations, delimit the math equations with $$.
    """

clean_text_human_prompt = """
    Given the text delimited by ```, clean it according to the rules.
    Return only the cleaned text, no other text or comments.
    ```
    {pdf_text}
    ```
    """

translate_text_system_prompt = """
Instruction / Persona:
你是一位專業譯者，擅長軟體類書籍的英翻中翻譯，翻譯風格偏重於：
- 忠實保留原文結構與內容，逐句逐段翻譯，不刪減、不意譯。
- 人名保持英文原文，例如 Leslie Lamport。
- 任何技術類專有名詞應保留英文原文，僅在首次出現時，在英文後面加上括號，括號內為中文翻譯，例如 "Prompt Engineering (提示詞工程)"。
- 程式碼、CLI 指令會以 ``` 包裹，僅翻譯程式碼中的註釋，不要翻譯程式碼中的程式碼。
- 翻譯必須讓中文讀者讀起來流暢自然，同時能對照英文概念。

Behavior Rules:
- 當使用者貼上英文段落時，一次完整翻譯整段，不可省略或跳段。
- 不要自行提問「是否要繼續翻譯下一段」，只翻譯使用者提供的內容。
- 翻譯結果要符合「技術書籍出版品質」，避免口語化或過度意譯。

Example:
- User input: ```The CAP theorem was initially a conjecture made by computer scientist Eric Brewer...```
- Assistant response: ```The CAP theorem 最初是電腦科學家 Eric Brewer 提出的猜想...```
    """

translate_text_human_prompt = """
    Given the text delimited by ```, translate it according to the rules.
    Return only the translated text, no other text or comments.
    ```
    {cleaned_text}
    ```
    """


def main():
    print("Loading PDF...")
    loader = PyPDFLoader("data/ai-system-evaluation-criteria.pdf")
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    selected_pages = docs[1:4]
    pdf_text_page_contents = [page.page_content for page in selected_pages]

    # LLM 1: Clean text extracted from
    gemini_2_5_flash = init_chat_model(
        model_provider="google_genai",
        model="gemini-2.5-flash",
        temperature=0.0,
    )
    clean_text_prompt = ChatPromptTemplate(
        [
            ("system", clean_text_system_prompt),
            ("user", clean_text_human_prompt),
        ]
    )
    clean_text_chain = clean_text_prompt | gemini_2_5_flash

    # LLM 2: Translate text
    gpt_4o = init_chat_model(
        model_provider="openai",
        model="gpt-4o",
        temperature=0.0,
    )
    translate_text_prompt = ChatPromptTemplate(
        [
            ("system", translate_text_system_prompt),
            ("user", translate_text_human_prompt),
        ]
    )
    translate_text_chain = translate_text_prompt | gpt_4o

    full_text_chain = clean_text_chain | translate_text_chain

    for page_content in pdf_text_page_contents:
        translated_text = full_text_chain.invoke({"pdf_text": page_content})
        print("-" * 50)
        print(translated_text.content)


if __name__ == "__main__":
    main()
