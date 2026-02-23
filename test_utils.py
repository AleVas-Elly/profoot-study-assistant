import json
import random
from langchain_core.prompts import PromptTemplate

import time
import math

import db_utils

def generate_mock_test(api_keys, book_source, chapter_docs, quotas, num_options, progress_callback=None):
    """
    Generates a multiple-choice quiz from textbook chunks.
    Questions are distributed across chapters proportionally. API keys are rotated
    automatically when rate limits are hit, with exponential backoff.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    BATCH_SIZE = 5
    all_questions = []
    current_key_idx = 0
    total_questions = sum(quotas.values())
    questions_generated_so_far = 0
    
    # Fun anatomy themed messages
    FUN_MSGS = [
        "🫀 Pumping data into the knowledge heart...",
        "🧠 Firing neurons to generate questions...",
        "🦴 Building the skeletal structure of your test...",
        "🫁 Deep breathing while reading the textbook...",
        "🔬 Dissecting the Dutch chapters...",
        "🩸 Circulating through the medical facts...",
        "💪 Strengthening your mental muscles...",
        "👁️ Observing the anatomical details..."
    ]

    def get_progress_data(pct, debug_msg):
        # Pick a fun message based on progress
        msg_idx = min(int(pct * len(FUN_MSGS)), len(FUN_MSGS)-1)
        return pct, FUN_MSGS[msg_idx], debug_msg

    if progress_callback:
        progress_callback(*get_progress_data(0.0, "Initializing API chain..."))

    def get_llm_chain(key):
        return [
            ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=key, temperature=0.3, timeout=60.0, max_retries=0),
            ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=key, temperature=0.3, timeout=60.0, max_retries=0),
            ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=key, temperature=0.3, timeout=60.0, max_retries=0)
        ]

    template = """
    You are an expert medical professor creating a rigorous multiple-choice exam for your anatomy class.
    
    Based STRICTLY on the provided Dutch textbook excerpts, generate exactly {batch_questions} multiple-choice questions.
    Each question must have exactly {num_options} options.
    
    CRITICAL: The questions, options, correct_answer, correct_explanation, and incorrect_explanations MUST all be translated and written in ENGLISH. 
    Do NOT write the test in Dutch, even though the source text is Dutch.
    
    {exclusion_list}
    
    For the correct answer, you must provide a brief paragraph (`correct_explanation`) explaining WHY it is the correct answer. You do not need to limit this explanation to the provided text; you may draw upon your general medical knowledge to provide a clear, comprehensive "why".
    For the incorrect answers, you must provide a dictionary (`incorrect_explanations`) where the keys are the incorrect options and the values are brief explanations of WHY they are wrong.
    
    For each question, you MUST provide the page number and a short snippet directly from the provided text that proves the `correct_answer`.
    These fields MUST be named `source_page` and `source_snippet`. If you can't find the exact page, write "Unknown Page". The `source_snippet` should be in the original Dutch to match the book.
    
    You MUST output valid JSON only. Do not include markdown blocks like ```json ... ```. Just the raw array exactly in this format:
    
    [
      {{
        "question": "What is the primary function of the heart according to the text?",
        "options": ["To pump blood", "To digest food", "To breathe air"],
        "correct_answer": "To pump blood",
        "correct_explanation": "The heart acts as a central pump that circulates oxygenated blood and nutrients throughout the body and removes metabolic waste.",
        "incorrect_explanations": {{
             "To digest food": "Digestion is primarily the function of the stomach and intestines.",
             "To breathe air": "Gas exchange and breathing are the primary functions of the lungs."
        }},
        "source_page": "Page 5",
        "source_snippet": "The heart is a central pump."
      }}
    ]
    
    Dutch Textbook Excerpts:
    {context}
    """

    def chapter_sort_key(ch):
        if not ch: return 999
        if ch.lower() in ("preface / intro", "inleiding"): return -1
        match = re.search(r'\d+', ch)
        return int(match.group()) if match else 999
        
    sorted_chapter_items = sorted(quotas.items(), key=lambda x: chapter_sort_key(x[0]))

    for chapter, num_questions in sorted_chapter_items:
        if num_questions <= 0:
            continue
            
        docs = chapter_docs.get(chapter, [])
        if not docs:
            continue
            
        past_questions = db_utils.get_past_questions(book_source, chapter, limit=20)
        exclusion_text = ""
        if past_questions:
            exclusion_text = "CRITICAL INSTRUCTION TO PREVENT DUPLICATES:\nDO NOT generate questions similar to these past questions:\n"
            for past_q in past_questions:
                exclusion_text += f"- {past_q}\n"
                
        num_batches = math.ceil(num_questions / BATCH_SIZE)
        chapter_questions_generated = 0
        
        for batch_idx in range(num_batches):
            questions_left = num_questions - chapter_questions_generated
            batch_questions = min(BATCH_SIZE, questions_left)
            
            # Select documents for this batch from this chapter
            sample_size = min(len(docs), 25 + (batch_questions * 2))
            batch_docs = random.sample(docs, sample_size)
            context_text = "\n\n".join([f"--- Page: {doc.metadata.get('page', 'Unknown')} ---\n{doc.page_content}" for doc in batch_docs])
    
            batch_success = False
            
            while not batch_success:
                current_key = api_keys[current_key_idx % len(api_keys)]
                llm_chain = get_llm_chain(current_key)
                
                if progress_callback:
                    pct = questions_generated_so_far / total_questions
                    debug_msg = f"API Key {current_key_idx % len(api_keys) + 1} | {chapter} Batch {batch_idx+1}/{num_batches}"
                    progress_callback(*get_progress_data(pct, debug_msg))
    
                prompt = PromptTemplate.from_template(template).format(
                    batch_questions=batch_questions, 
                    num_options=num_options,
                    context=context_text,
                    exclusion_list=exclusion_text
                )
                
                # Attempt with current key through all models
                for llm_idx, llm in enumerate(llm_chain):
                    try:
                        response = llm.invoke(prompt)
                        text = response.content
                        if isinstance(text, list):
                            text = "".join([b["text"] if isinstance(b, dict) and "text" in b else str(b) for b in text])
                        
                        text = str(text).strip()
                        if text.startswith("```json"): text = text[7:]
                        if text.startswith("```"): text = text[3:]
                        if text.endswith("```"): text = text[:-3]
                        
                        try:
                            parsed_json = json.loads(text.strip())
                        except json.JSONDecodeError as je:
                            print(f"JSON Parsing Error on Model {llm_idx+1}: {je}\nRaw LLM Output:\n{text[:500]}...")
                            raise je # Re-raise to trigger the retry logic
                        
                        # Validate exactly the requested amount
                        parsed_json = parsed_json[:batch_questions]
                        
                        # Inject chapter info
                        for q in parsed_json:
                            q['chapter'] = chapter
                        
                        all_questions.extend(parsed_json)
                        chapter_questions_generated += len(parsed_json)
                        questions_generated_so_far += len(parsed_json)
                        
                        # Save newly generated questions to avoid repeats in the future
                        db_utils.save_past_questions(book_source, chapter, parsed_json)
                        
                        batch_success = True
                        break
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            print(f"Key {current_key_idx % len(api_keys) + 1} Model {llm_idx+1} rate limited. Trying next...")
                            continue
                        else:
                            print(f"Unexpected error on Key {current_key_idx % len(api_keys) + 1} Model {llm_idx+1}: {str(e)[:200]}")
                            # Don't break here, let it try the next fallback model in the chain
                            continue
                
                if not batch_success:
                    # All models failed for this key
                    current_key_idx += 1
                    if current_key_idx % len(api_keys) == 0:
                        # We cycled through ALL keys. Apply backoff.
                        wait_time = 20
                        if progress_callback:
                            pct = questions_generated_so_far / total_questions
                            progress_callback(*get_progress_data(pct, f"Cooldown: Waiting {wait_time}s..."))
                        time.sleep(wait_time)
                    else:
                        if progress_callback:
                            pct = questions_generated_so_far / total_questions
                            progress_callback(*get_progress_data(pct, f"Rotating to Key {current_key_idx % len(api_keys) + 1}..."))

    if progress_callback:
        progress_callback(1.0, "🏁 Test Generated!", "Success")
    return all_questions
