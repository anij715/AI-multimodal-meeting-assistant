import pandas as pd
import ast
import os
import requests
import json
import torch
import re
import traceback
import logging
import argparse
from io import StringIO
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    AutoProcessor, LlavaForConditionalGeneration, GenerationConfig
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from openpyxl.drawing.image import Image as OpenpyxlImage
from openpyxl.utils import get_column_letter
import docx
from docx.opc.exceptions import PackageNotFoundError
from contextlib import redirect_stdout
import io
import functools

# --- CONFIGURATION & SETUP ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_EXCEL_PATH = "/home3/sharmarz/Project-test-pipelines/Benchmark/questions.xlsx"
ANALYSIS_OUTPUT_EXCEL_PATH = "/home3/sharmarz/Project-test-pipelines/Output/analysis_results.xlsx"
QA_OUTPUT_EXCEL_PATH = "/home3/sharmarz/Project-test-pipelines/Output/qa_results.xlsx"
DIRECT_QA_OUTPUT_EXCEL_PATH = "/home3/sharmarz/Project-test-pipelines/Output/direct_qa_results.xlsx"
FINAL_TRANSCRIPT_QA_PATH = "/home3/sharmarz/Project-test-pipelines/Output/final_transcript_qa_results.xlsx"
FINAL_DIRECT_QA_PATH = "/home3/sharmarz/Project-test-pipelines/Output/final_direct_qa_results.xlsx"

OUTPUT_DIR = "/home3/sharmarz/Project-test-pipelines/Output"
CHART_IMG_DIR = os.path.join(OUTPUT_DIR, "Initial_charts")
TEXT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
VLM_MODEL = "llava-hf/llava-1.5-7b-hf"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_IMG_DIR, exist_ok=True)

# Global variables for models
text_model, text_tokenizer, text_device = None, None, None
vlm_model, vlm_processor = None, None

# --- Data Loading and Initial Setup ---
try:
    csv_url = "https://mailuc-my.sharepoint.com/:x:/g/personal/sharmarz_mail_uc_edu/Ef5POYmr8M1PoXaiJbh-QlEBLkKKf4K8AHwhm--qsSnixA?download=1"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(csv_url, headers=headers, timeout=90)
    response.raise_for_status()
    try:
        csv_text_content = response.content.decode('utf-8')
        logging.info("Successfully decoded content as UTF-8.")
    except UnicodeDecodeError:
        csv_text_content = response.content.decode('latin1')
        logging.info("Decoded content as latin1 after UTF-8 failure.")
    df = pd.read_csv(StringIO(csv_text_content), dtype={10: str})
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df.dropna(subset=['revenue', 'budget', 'release_date'], inplace=True)
    df = df[(df['revenue'] > 0) & (df['budget'] > 0)]
    schema = str(df.dtypes)
    statistical_summary = df.describe().to_string()
except Exception as e:
    logging.fatal(f"FATAL: Could not download or process base CSV file for schema. Error: {e}")
    exit()

# === HELPER FUNCTIONS ===

def load_transcript_from_file(file_path):
    if pd.isna(file_path) or not file_path or not os.path.exists(file_path): 
        return None
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except (PackageNotFoundError, IOError) as e:
        logging.error(f"Could not read docx file {file_path}. Error: {e}")
        return None

def load_chart_code_from_file(file_path):
    if not file_path or not os.path.exists(file_path): return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [f.read()]
    except (IOError, UnicodeDecodeError) as e:
        logging.error(f"Could not read code file {file_path}. Error: {e}")
        return []


def parse_qa_response(response_text):
    """
    Parses the model's response to extract the answer and rationale,
    handling multiple common formats.
    Returns a tuple (answer, rationale).
    """
    if not isinstance(response_text, str):
        response_text = str(response_text)
    response_text = response_text.strip()
    answer = "N/A"
    rationale = response_text

    try:
        # Pattern 1: Explicit "Answer: [A-D]" (potentially at the start or on its own line)
        answer_match = re.search(r"Answer:\s*([A-D])", response_text, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).upper()
            # Clean up rationale: remove the "Answer: ..." part and the "Rationale:" prefix
            temp_rationale = re.sub(r"Answer:\s*[A-D]\s*", "", response_text, count=1, flags=re.IGNORECASE).strip()
            rationale_match = re.search(r"Rationale:(.*)", temp_rationale, re.IGNORECASE | re.DOTALL)
            if rationale_match:
                rationale = rationale_match.group(1).strip()
            else:
                rationale = temp_rationale
            return answer, rationale

        # Pattern 2: Phrase like "the correct answer is [A-D]"
        answer_match = re.search(r"the correct answer is\s+([A-D])", response_text, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).upper()
            # The full text is the rationale in this case
            return answer, response_text

        # Pattern 3: The entire response is just a single letter "C"
        if re.fullmatch(r"\s*([A-D])\s*", response_text, re.IGNORECASE):
            answer = response_text.strip().upper()
            rationale = ""  # No rationale was provided
            return answer, rationale

        # Pattern 4: Answer in the format "C)"
        # Searches for a letter (A-D) at the beginning of a word, followed by a parenthesis.
        answer_match = re.search(r"\b([A-D])\)", response_text, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).upper()
            # The full text is the rationale
            return answer, response_text

        # If no specific pattern is found, return the default
        return "N/A", response_text

    except Exception as e:
        logging.error(f"Failed to parse Q&A response with error: {e}")
        return "PARSE_ERROR", response_text

def parse_qa_response_1(response_text):
    """
    Parses the model's response to extract the answer and rationale.
    Returns a tuple (answer, rationale).
    """
    try:
        # Find the answer (a single letter)
        answer_match = re.search(r"Answer:\s*([A-Da-d])", response_text, re.IGNORECASE)
        answer = answer_match.group(1).upper() if answer_match else "N/A"
        
        # Find the rationale
        rationale_match = re.search(r"Rationale:(.*)", response_text, re.IGNORECASE | re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else response_text

        # If we found an answer, don't include the "Answer: A" part in the rationale
        if answer != "N/A":
             return answer, rationale
        # If we didn't find a clear answer, the whole text is the rationale
        else:
             return "N/A", response_text

    except Exception as e:
        logging.error(f"Failed to parse Q&A response: {e}")
        return "PARSE_ERROR", response_text

def sanitize_path_to_filename(path_str):
    """Converts a file path into a safe, unique string for use as a filename base."""
    if pd.isna(path_str) or not path_str:
        return "invalid_path"
    return re.sub(r'[/\\.]', '_', path_str).replace('__py', '')

# === CORE AI FUNCTIONS ===

def initialize_text_model_once(model_name):
    global text_model, text_tokenizer, text_device
    if text_model is not None: return
    logging.info(f"Initializing text model: {model_name}")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    text_tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
    text_device = text_model.device
    text_model.eval()
    logging.info("Text model initialized.")

def initialize_vlm_once(model_name):
    global vlm_model, vlm_processor
    if vlm_model is not None: return
    logging.info(f"Initializing VLM model: {model_name}")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    vlm_processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    vlm_model = LlavaForConditionalGeneration.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto", torch_dtype=torch.float16)
    vlm_model.eval()
    logging.info("VLM initialized.")

def generate_text_locally(prompt_text, max_new_tokens=3072, temperature=0.2):
    global text_model, text_tokenizer, text_device
    if text_model is None: raise EnvironmentError("Text model not initialized.")
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature, pad_token_id=text_tokenizer.eos_token_id)
    inputs = text_tokenizer(f"<s>[INST] {prompt_text} [/INST]", return_tensors="pt").to(text_device)
    with torch.no_grad():
        output_sequences = text_model.generate(**inputs, generation_config=generation_config)
    return text_tokenizer.decode(output_sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def generate_from_image_and_text(prompt_text, image_paths, max_new_tokens=1024):
    global vlm_model, vlm_processor
    if vlm_model is None: raise EnvironmentError("VLM not initialized.")
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=True)
    try:
        images = []
        if image_paths:
            images = [Image.open(path).convert('RGB') for path in image_paths if os.path.exists(path)]
            if not images:
                logging.warning("VLM generation warning: No valid images found for provided paths.")

        prompt = f"USER: {''.join(['<image>\\n' for _ in images])}{prompt_text}\nASSISTANT:"
        inputs = vlm_processor(text=prompt, images=images if images else None, return_tensors="pt").to(vlm_model.device)
        
        with torch.no_grad():
            output = vlm_model.generate(**inputs, generation_config=generation_config)
        
        # Cleaning the output to get only the assistant's response
        decoded_output = vlm_processor.decode(output[0], skip_special_tokens=True)
        return decoded_output.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in decoded_output else decoded_output

    except Exception as e:
        logging.error(f"Error in VLM generation: {e}\n{traceback.format_exc()}")
        return f"Error: {e}"

def generate_initial_chart_images(code_strings, base_filename):
    """
    Executes a code block that may generate multiple plots and saves each one.
    It works by temporarily replacing `plt.show()` to intercept plot generation.
    """
    output_paths = []
    code_string = code_strings[0] if code_strings else ""
    if not code_string:
        return []

    class _SaveShow:
        def __init__(self, base_path):
            self.base_path = base_path
            self.count = 0
            self.filenames = []
            self.original_show = plt.show

        def __enter__(self):
            def save_and_show_wrapper(*args, **kwargs):
                filename = f"{self.base_path}_{self.count}.png"
                output_path = os.path.join(CHART_IMG_DIR, filename)
                plt.savefig(output_path, bbox_inches='tight')
                plt.close('all')
                self.filenames.append(filename)
                self.count += 1
            
            # Using functools to make the wrapper look like the original function, avoiding signature errors.
            functools.update_wrapper(save_and_show_wrapper, self.original_show)
            plt.show = save_and_show_wrapper
            return self
        
        def save_if_any_figures_left(self):
            """If a figure exists but show() wasn't called, save it."""
            if plt.get_fignums():
                plt.show()

        def __exit__(self, exc_type, exc_val, exc_tb):
            plt.show = self.original_show

    try:
        with _SaveShow(base_filename) as saver:
            local_env = {"pd": pd, "plt": plt, "sns": sns, "ast": ast, "requests": requests, "StringIO": StringIO, "df": df}
            with redirect_stdout(io.StringIO()):
                exec(code_string, local_env)
            
            # Explicitly saving any figure that might have been created but not shown.
            saver.save_if_any_figures_left()
            output_paths = saver.filenames
    except Exception as e:
        logging.error(f"Chart exec failed for {base_filename}. Error: {e}")
        plt.close('all')

    return output_paths

# === PIPELINE FUNCTIONS ===

def analyze_initial_charts(mode, image_paths, code_strings, schema_str, stats_summary):
    num_items = len(image_paths if image_paths else code_strings)
    
    desc_prompt = f"You are given {num_items} data visualizations. Describe each visualization in a detailed paragraph. Provide a holistic summary at the end."
    heuristic_prompt = "Analyze the provided chart(s). Provide a single-sentence heuristic description summarizing all the charts."
    spec_prompt = "Generate a JSON formatted specification for each of the charts provided."

    text_context = f"Chart Code:\n{''.join(code_strings)}\nData Schema:\n{schema_str}\nStats:\n{stats_summary}"
    
    # Determine the generation function based on the mode
    if mode == 'image_only':
        gen_func = functools.partial(generate_from_image_and_text, image_paths=image_paths)
        context = ""
    elif mode == 'text_only':
        gen_func = generate_text_locally
        context = f"{text_context}\nTask: "
    elif mode == 'llava_text_only':
        gen_func = functools.partial(generate_from_image_and_text, image_paths=[])
        context = f"{text_context}\nTask: "
    elif mode == 'hybrid':
        gen_func = functools.partial(generate_from_image_and_text, image_paths=image_paths)
        context = f"Use the images and text.\n{text_context}\nTask: "
    else:
        return {"prose_description": "Invalid mode.", "heuristic_description": "Invalid mode.", "chart_specification": "Invalid mode."}

    # Make separate calls for each piece of information
    prose_desc = gen_func(f"{context}{desc_prompt}")
    heuristic_desc = gen_func(f"{context}{heuristic_prompt}")
    chart_spec = gen_func(f"{context}{spec_prompt}")

    return {"prose_description": prose_desc, "heuristic_description": heuristic_desc, "chart_specification": chart_spec}

def extract_feedback(context, transcript_str, schema_str, stats_summary):
    prompt = f'''
You are an AI assistant analyzing user feedback on a set of chart/s. Use all the provided context to perform your task.

**CONTEXT 1: Analysis of the Initial Charts (Generated via {context['mode']} method)**
- Prose Description: {context['prose_description']}
- Heuristic Description: {context['heuristic_description']}
- Chart Specification: {context['chart_specification']}

**CONTEXT 2: Ground Truth Data**
- Data Schema: {schema_str}
- Statistical Summary:
{stats_summary}

**CONTEXT 3: User Conversation**
- Transcript: {transcript_str}

**TASK:**
Based on ALL of the above context, extract the unique critiques and suggestions from the user transcript. List them as a bulleted list.
'''
    return generate_text_locally(prompt, max_new_tokens=1024)

def style_instruction_from_model(feedback):
    prompt = f'''
Based on the feedback below, suggest the best way to visualize the data in a single, concise instruction for one new chart that combines the best elements and fixes.\n\nFeedback:\n{feedback}\n\nOne-line instruction for a single new chart:
'''
    return generate_text_locally(prompt, max_new_tokens=100)

def handle_question(mode, context, feedback, transcript, question_string):
    """Handles a follow-up question using the full context of a specific pipeline run."""
    
    full_prompt = f"""You are an expert AI assistant. Your mission is to answer the user's question by synthesizing information from ALL the provided sources.

### Source 1: The Original User Conversation (Transcript)
{transcript}

### Source 2: AI's Extracted Summary of Feedback (for this '{mode}' mode)
{feedback}

### Source 3: AI's Analysis of the Initial Chart (for this '{mode}' mode)
- Prose Description: {context.get('prose_description', 'N/A')}
- Heuristic Description: {context.get('heuristic_description', 'N/A')}
- Chart Specification: {context.get('chart_specification', 'N/A')}

### Source 4: The User's Question
{question_string}

YOUR TASK:
Carefully read the User's Question, choose the single best option (A, B, C, or D) by synthesising all of the provided sources (through 1 to 3). Then, provide a detailed rationale for your choice, citing your sources.
Your response MUST follow this format exactly:
Answer: [Your chosen letter]
Rationale: [Your detailed explanation]
"""
    
    image_paths = [os.path.join(CHART_IMG_DIR, p.strip()) for p in context.get('chart_paths', [])]
    
    if mode in ['image_only', 'hybrid', 'llava_text_only']:
        image_list = image_paths if mode != 'llava_text_only' else []
        return generate_from_image_and_text(full_prompt, image_list)
    else: # text_only (Mixtral)
        return generate_text_locally(full_prompt)

def handle_direct_question(mode, context, question_string):
    """Handles a direct question using only the initial chart analysis."""
    
    prompt = f"""You are a data analysis assistant. Your task is to answer the question based *only* on the provided chart analysis and images.

### Source 1: AI's Analysis of the Chart(s)
- Prose Description: {context.get('prose_description', 'Analysis not available.')}
- Heuristic Description: {context.get('heuristic_description', 'Analysis not available.')}
- Chart Specification: {context.get('chart_specification', 'Analysis not available.')}

### The User's Question
{question_string}

YOUR TASK:
First, choose the single best option (A, B, C, or D). Then, provide a detailed rationale for your choice.
Your response MUST follow this format exactly:
Answer: [Your chosen letter]
Rationale: [Your detailed explanation]
"""
    image_paths = [os.path.join(CHART_IMG_DIR, p.strip()) for p in context.get('chart_paths', [])]

    if mode in ['image_only', 'hybrid', 'llava_text_only']:
        image_list = image_paths if mode != 'llava_text_only' else []
        return generate_from_image_and_text(prompt, image_list)
    else: # text_only (Mixtral)
        # Text-only mode needs the text context for the chart, which isn't available here.
        # So we can't answer, but we can pass an empty prompt to get a "can't answer" response.
        return generate_text_locally(prompt)
    
def write_excel_file(df_dict, output_path, image_cols_map):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                if image_cols_map:
                    worksheet = writer.sheets[sheet_name]
                    for col_name, col_letter in image_cols_map.items():
                        if col_name not in df.columns: continue
                        worksheet.column_dimensions[col_letter].width = 30
                        for index, row in df.iterrows():
                            row_number = index + 2
                            worksheet.row_dimensions[row_number].height = 120
                            filename_str = row.get(col_name, '')
                            if pd.isna(filename_str) or not filename_str: continue
                            img_path = os.path.join(CHART_IMG_DIR, filename_str.split(',')[0].strip())
                            if os.path.exists(img_path):
                                try:
                                    img = OpenpyxlImage(img_path)
                                    img.height, img.width = 150, 180
                                    worksheet.add_image(img, f"{col_letter}{row_number}")
                                except Exception as e:
                                    logging.error(f"Could not insert image {img_path}. Error: {e}")

# === MAIN BATCH PROCESSING EXECUTION ===

def main():
    try:
        initialize_text_model_once(TEXT_MODEL)
        initialize_vlm_once(VLM_MODEL)
    except Exception as e:
        logging.fatal(f"FATAL: Could not initialize models. Error: {e}")
        return

    try:
        xls = pd.ExcelFile(INPUT_EXCEL_PATH)
        tasks_df = pd.read_excel(xls, 'Questions-Transcript-Charts')
        tasks_df = tasks_df.dropna(subset=['question'])
    except (FileNotFoundError, ValueError) as e:
        logging.fatal(f"FATAL: Input file not found or sheet missing at {INPUT_EXCEL_PATH}. Error: {e}")
        return

    # --- STAGE 1: ANALYSIS ---
    logging.info("--- Starting Stage 1: Analysis ---")
    analysis_columns = ["transcript_name", "transcript_path", "chart_code_path", "initial_generated_chart_img", "prose_description", "heuristic_description", "chart_specification", "extracted_feedback", "suggested_chart_style_instruction"]
    analysis_results = {mode: [] for mode in ['image_only', 'text_only', 'llava_text_only', 'hybrid']}
    
    artifacts_df = tasks_df[['chart_code_path', 'transcript_name', 'transcript_path']].copy().drop_duplicates()
    all_artifacts = artifacts_df.to_dict('records')

    for artifact in all_artifacts:
        code_path = artifact['chart_code_path']
        logging.info(f"Analyzing artifact: {code_path}")
        
        transcript = load_transcript_from_file(artifact.get('transcript_path'))
        base_chart_codes = load_chart_code_from_file(artifact['chart_code_path'])

        if not base_chart_codes:
            logging.warning(f"SKIPPING artifact {artifact.get('transcript_name', 'N/A')}: No code found.")
            continue
        
        initial_base_filename = f"initial_{sanitize_path_to_filename(code_path)}"
        initial_png_filenames = generate_initial_chart_images(base_chart_codes, initial_base_filename)
        for mode in analysis_results.keys():
            logging.info(f"  Running mode: {mode}")
            context = analyze_initial_charts(mode, [os.path.join(CHART_IMG_DIR, f) for f in initial_png_filenames], base_chart_codes, schema, statistical_summary)
            context['mode'] = mode
            context['chart_paths'] = initial_png_filenames
            
            feedback = extract_feedback(context, transcript, schema, statistical_summary) if transcript else "N/A"
            style_instruction = style_instruction_from_model(feedback) if feedback != "N/A" else "N/A"
            
            analysis_results[mode].append({
                "transcript_name": artifact.get('transcript_name', 'N/A'), "transcript_path": artifact.get('transcript_path', 'N/A'),
                "chart_code_path": artifact['chart_code_path'], "initial_generated_chart_img": ",".join(initial_png_filenames),
                "prose_description": context['prose_description'],
                "heuristic_description": context['heuristic_description'],
                "chart_specification": context['chart_specification'],
                "extracted_feedback": feedback,
                "suggested_chart_style_instruction": style_instruction,
            })

    analysis_dfs = {mode: pd.DataFrame(data, columns=analysis_columns) for mode, data in analysis_results.items()}
    write_excel_file(analysis_dfs, ANALYSIS_OUTPUT_EXCEL_PATH, {'initial_generated_chart_img': 'D'})
    logging.info(f"--- Stage 1 Complete: Analysis results saved to {ANALYSIS_OUTPUT_EXCEL_PATH} ---")

    # --- STAGE 2: TRANSCRIPT-BASED Q&A ---
    logging.info("--- Starting Stage 2: Transcript-Based Q&A ---")
    qa_columns = ["transcript_name", "chart_code_path", "question_asked", "option_a", "option_b", "option_c", "option_d", "pipeline_answer", "pipeline_rationale"]
    qa_results = {mode: [] for mode in analysis_results.keys()}
    
    transcript_analysis_cache = {}
    for mode, df_mode in analysis_dfs.items():
        # Filter for rows that actually have a transcript path to create a valid index
        transcript_df = df_mode.dropna(subset=['transcript_path'])
        if not transcript_df.empty:
            transcript_analysis_cache[mode] = transcript_df.set_index(['transcript_path', 'chart_code_path'])

    for index, task in tasks_df.iterrows():
        logging.info(f"Answering transcript question {index+1}/{len(tasks_df)} for artifact: {task['transcript_name']}")
        for mode in qa_results.keys():
            if mode not in transcript_analysis_cache: continue
            try:
                # Use a single value for the lookup, not a tuple, to avoid the error
                cached_row = transcript_analysis_cache[mode].loc[(task['transcript_path'], task['chart_code_path'])]
                # Ensure we handle cases where multiple rows might be returned by taking the first one
                if isinstance(cached_row, pd.DataFrame):
                    cached_row = cached_row.iloc[0]

                context = {
                    "mode": mode, "prose_description": cached_row["prose_description"],
                    "heuristic_description": cached_row["heuristic_description"], "chart_specification": cached_row["chart_specification"],
                    "chart_paths": cached_row["initial_generated_chart_img"].split(',') if pd.notna(cached_row["initial_generated_chart_img"]) else []
                }
                feedback = cached_row['extracted_feedback']
                transcript = load_transcript_from_file(task['transcript_path'])
                
                question_string = f"{task['question']}\nA) {task['option_a']}\nB) {task['option_b']}\nC) {task['option_c']}\nD) {task['option_d']}"
                
                pipeline_response_raw = handle_question(mode, context, feedback, transcript, question_string)
                answer, rationale = parse_qa_response(pipeline_response_raw)
                
                qa_results[mode].append({
                    "transcript_name": task['transcript_name'], "chart_code_path": task['chart_code_path'],
                    "question_asked": task['question'], "option_a": task['option_a'], "option_b": task['option_b'],
                    "option_c": task['option_c'], "option_d": task['option_d'],
                    "pipeline_answer": answer, "pipeline_rationale": rationale
                })
            except KeyError:
                logging.error(f"Could not find analysis context for {task['transcript_name']} in mode {mode}. Skipping Q&A.")
                continue

    qa_dfs = {mode: pd.DataFrame(data, columns=qa_columns) for mode, data in qa_results.items()}
    write_excel_file(qa_dfs, QA_OUTPUT_EXCEL_PATH, None)
    logging.info(f"--- Stage 2 Complete: Q&A results saved to {QA_OUTPUT_EXCEL_PATH} ---")


    # --- FINAL STAGE: MERGE for FINAL TRANSCRIPT-BASED OUTPUT ---
    logging.info("--- Starting Final Stage: Merging Results into a Single File ---")
    
    final_transcript_dfs = {}
    for mode in analysis_results.keys():
        if mode in analysis_dfs and mode in qa_dfs and not analysis_dfs[mode].empty and not qa_dfs[mode].empty:
            
            analysis_cols_to_keep = [
                "transcript_name", "chart_code_path", "transcript_path", "initial_generated_chart_img",
                "prose_description", "heuristic_description", "chart_specification",
                "extracted_feedback", "suggested_chart_style_instruction"
            ]
            qa_cols_to_keep = [
                "transcript_name", "chart_code_path", "question_asked", "option_a",
                "option_b", "option_c", "option_d", "pipeline_answer", "pipeline_rationale"
            ]

            left_df = analysis_dfs[mode][analysis_cols_to_keep]
            right_df = qa_dfs[mode][qa_cols_to_keep]
            
            final_transcript_dfs[mode] = pd.merge(
                left_df,
                right_df,
                on=['transcript_name', 'chart_code_path']
            )

    write_excel_file(final_transcript_dfs, FINAL_TRANSCRIPT_QA_PATH, {'initial_generated_chart_img': 'D'})
    logging.info(f"--- Pipeline Complete: Final transcript-based results saved to {FINAL_TRANSCRIPT_QA_PATH} ---")

if __name__ == '__main__':
    main()