import ast
import difflib
import traceback

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

import uuid

import time
# from langfuse.openai import openai
# from langfuse import Langfuse
from langchain.prompts import ChatPromptTemplate
import subprocess
import os

# from langfuse.langchain import CallbackHandler

from colorama import init, Fore
from loguru import logger

# Initialize colorama
init(autoreset=True)
is_ollama = False

open_ai_key = ""
os.environ["OPENAI_API_KEY"] = open_ai_key

# Add your keys if needed
# langfuse = Langfuse()
# langfuse_handler = CallbackHandler()
from .utils import Config

# if not is_ollama:
#     llm_openai = ChatOpenAI(model="gpt-4o", temperature=0.2)
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
#     code_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
# else:
#     llm = ChatOllama(
#         model="deepseek-coder-v2", temperature=0.2, base_url="http://localhost:11434/"
#     )
#     llm_openai = ChatOllama(
#         model="deepseek-coder-v2", temperature=0.2, base_url="http://localhost:11434/"
#     )
#     code_llm = ChatOllama(
#         model="deepseek-coder-v2", temperature=0.2, base_url="http://localhost:11434/"
#     )

сode_bank = []
code_llm = None
llm = None
llm_openai = None
config:Config | None = None


def shorten_string_middle(text, max_length):
    """
    Shorten a string from the middle and replace the removed part with an ellipsis.

    Args:
        text (str): The original string to be shortened.
        max_length (int): The maximum allowed length of the resulting string, including the ellipsis.

    Returns:
        str: The shortened string with an ellipsis if it exceeds the max_length, otherwise the original string.

    Raises:
        ValueError: If max_length is less than 5, as it is not practical to display the ellipsis and any characters.
    """
    if max_length < 5:
        raise ValueError(
            "max_length must be at least 5 to accommodate the ellipsis and at least one character from each end."
        )

    if len(text) <= max_length:
        return text

    # Calculate the number of characters to show from the start and end
    num_chars_each_side = (max_length - 3) // 2

    # Handle odd number of characters to display
    start_part = text[:num_chars_each_side]
    end_part = text[-(max_length - 3 - num_chars_each_side) :]

    # Construct the shortened string
    return f"{start_part}...{end_part}"


def task_complexity_check(task, main_task):
    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert machine learning engineer and kaggler. 
        You will be given a sub-task, and the main task for context. (Assume the sub-task is the main task if no main task is given)

        Your one and only goal is to determine if the task in hand is meant for multi-step reasoning or can be answered immediately using the information
        Your answers should be directed about EDA topic.

        Here are some tips that will help you make a decision:

        1) Always return False if it's the main task, unless its very very obvious trivia type of question
        2) If it's not the main task, you should really be conservative in splitting it(returning False) unless you really think splitting it would give added benefits. So most times you'll end up returning True, unless obviously its the main task, then you mostly return False.
        3) If the task is about executing a small bunch of code, return True
        4) If the main task and the task is the same return True.

        Then, return True if want to split the task, else return False. 
        """,
            ),
            ("user", "SUB - Task Description: {input}  Main Task - {main_task}"),
        ]
    )

    # Combine the prompt and the LLM in a chain, add a string output parser
    chain = prompt | llm

    # Run the chain with an example input

    result = chain.invoke(
        {"input": task, "main_task": main_task},
        # config={"callbacks": [langfuse_handler]},
    )
    # logger.info(result)
    return result.content


def perform_task_python(
    task,
    previous_answers="No Previous Answers",
    main_task_context="This is the main task",
):
    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """      
        You will be given a sub-task related to python, reference to the main task, and trustable data context. 

        Your task is to do the sub-task continuing previous code! Use any other information only for reference.
        Add all needed imports.

        You will also be given the answers for the other subtasks for reference. If previous answer is defined, you should to merge all previous codes together!
        If you need to split the data use 80/20 train/test. 
        If you need to find images, don't use os.listdir, instead use os.walk.

        For deep learning task You should use pytorch, pytorch lightning and albumentations.
        Do not use tensorflow, keras. 

        You are kaggle master, you prefer xgboost over random forest. 

        USE shap package to identify feature importance.

        Perform the task and return the answer as python code!
        YOU SHOULD BE VERY CAUTIOUS ABOUT CODE, YOU NEED TO SOLVE TASK WITH PREVIOUS CODE, DO NOT OVERWRITE PREVIOUS SOLUTION COMPLETELY. 
        ALL TRAINING PATH OR TESTING PATHS ARE THE SAME FROM PREVIOUS CODE.
        FULL CODE REQUIRED, CODE SHOULD BE EXECUTABLE!
        """,
            ),
            (
                "user",
                "SUB-TASK to do(Do not perform any other TASK apart from this!!!!): {input} \n\nMain Context(don't do the other tasks) : {main_task_context} \n\nPrevious code: {previous_answers}  ",
            ),
        ]
    )

    # Combine the prompt and the LLM in a chain, add a string output parser
    chain = prompt | code_llm

    # Run the chain with an example input

    result = chain.invoke(
        {
            "input": task,
            "main_task_context": main_task_context,
            "previous_answers": previous_answers,
        },
        # config={"callbacks": [langfuse_handler]},
    )
    # logger.info(result)
    return result.content


def finetune_code(task, code, error="There is no error"):
    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You will be given a task and a bunch of python code written to find a trustable answer, 

        Your task is to convert these lines of code into executable code, which returns the required answer of the task as output (logger.info). 
        It's okay to be concise, but the answer needs the be there. Do not fix, error, like a Junior, with try except, or if
        there is no variable or no column, do not add if else with permanent exit. Goal of the code is to PRODUCE RESULT!
        If there is no Module or import error, install it with pip, use os.command or similar.
        Never ever return multiple arguments. If you have multiple variables as answers, combine them in a meaningful way.
        Don't forget to add logger.infos.

        CUDA IS REQUIRED for pytorch, keras, pytorch lightning. 

        CODE SHOULD BE EFFICIENT AND OPTIMIZED. 

        FULL CODE REQUIRED, NO OTHER TEXTS, CODE SHOULD BE EXECUTABLE!
        If you need to find images, don't use os.listdir, instead use os.walk.
        If you have KeyError: 'feature1', or some similar error, you didn't create any features, so create them, according to data.
        Make sure to only return code as exactly what you return will be executed in a python compiler.

        RETURN ONLY EXECUTABLE CODE!
        """,
            ),
            ("user", "main task : {task} \n\n code : {code} \n\n Error: {error}"),
        ]
    )

    # Combine the prompt and the LLM in a chain, add a string output parser
    chain = prompt | code_llm

    # Run the chain with an example input

    result = chain.invoke(
        {"task": task, "code": code, "error": error},
        # config={"callbacks": [langfuse_handler]},
    )
    # logger.info(result)
    return result.content


def compare_with_previous_code(new_code, previous_code):
    tree1 = ast.parse(new_code)
    tree2 = ast.parse(previous_code)

    # Convert AST to strings (or other comparable format)
    tree_str1 = ast.dump(tree1)
    tree_str2 = ast.dump(tree2)

    # Use SequenceMatcher on the dumped AST strings
    matcher = difflib.SequenceMatcher(None, tree_str1, tree_str2)
    similarity = matcher.ratio()

    return similarity > 0.95

    # return fuzz.token_set_ratio(new_code, previous_code) > 97


def code_executor(code):
    script_path = config.workspace_dir / "code" / "gen_code.py"
    with open(script_path, "w") as file:
        file.write(code)
    process = subprocess.run(["python3", str(script_path)], capture_output=True, text=True)

    return {
        "output": shorten_string_middle(process.stdout, 70000),
        "errors": shorten_string_middle(process.stderr, 10000),
    }


def generate_code_and_execute(task, previous_answers="No Previous Answers"):
    if len(previous_answers) > 10000:
        previous_answers = shorten_string_middle(previous_answers, 10000)
    answer_text = perform_task_python(
        task, previous_answers="\n".join(сode_bank), main_task_context=previous_answers
    )
    answer_code = finetune_code(task, answer_text)
    time.sleep(0.5)
    # logger.info(answer_code)
    is_code_executed = False
    error = ""
    tries = 0

    retry_count = 0
    while not is_code_executed:
        try:
            if error != "":
                answer_code = finetune_code(task, answer_text, error=error)
                time.sleep(0.5)
            # Make sure that code is not the same
            try:
                is_code_not_new = compare_with_previous_code(
                    new_code=answer_code.replace("```", "").replace("python", ""),
                    previous_code="\n".join(сode_bank),
                )
                if is_code_not_new:
                    logger.info(Fore.RED + "CODES ARE THE SAME")
                    return answer_code, previous_answers
            except:
                logger.info()
            logger.info(Fore.BLUE + "RUNNING CODE")
            code_result = code_executor(
                answer_code.replace("```", "").replace("python", "")
            )
            logger.info(Fore.BLUE + "FINISHED RUNNING CODE")
            logger.info(f"\nOutput code result:\n{code_result['output']}")
            logger.info(f"Output code errors:\n{code_result['errors']}")
            if (
                code_result["errors"] == ""
                and len(code_result["output"]) > 2
                or "error" not in code_result["errors"].lower()
            ):
                is_code_executed = True
            else:
                error = code_result["errors"]
                if len(error) > 10000:
                    error = shorten_string_middle(error, 10000)
                logger.info(Fore.RED + "Error RUNNING CODE\n" + error)
            time.sleep(0.5)
        except (BaseException, Exception):
            error = traceback.format_exc()
            if "Invalid 'messages[1].content': string too long" not in error:
                error = shorten_string_middle(traceback.format_exc(), 10000)
            else:
                logger.info(task, answer_text, error)
            logger.info(Fore.RED + "Error RUNNING CODE\n" + error)
            time.sleep(0.5)
        tries += 1
        retry_count += 1
        if tries > 5:
            tries = 0
            answer_text = perform_task_python(
                task,
                previous_answers="\n".join(сode_bank),
                main_task_context=previous_answers,
            )
            error = ""
        if retry_count > 10:
            raise "Something wrong with the task"
    if len(code_result["output"]) > 16000:
        code_result["output"] = shorten_string_middle(code_result["output"], 10000)
    logger.info(Fore.GREEN + "Final Code Result")
    logger.info(Fore.YELLOW + code_result["output"])
    return str(code_result["output"]), answer_code


def checks_generation(task):
    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You will be given a question. Assumming that there has been an answer generated for the question, what are the checks
        we can have in place to make sure that the answer is right beyond any doubt using python code

        Definition of a test - A scientifically provable question that is as specific as possible. 

        A test can be ONLY be proved using 3 ways:

        Trustable ways - If answer can be validated using these ways, pick it without doubt

        PLEASE OUTPUT THE MINIMUM QUANTITY OF CHECKS. THE IDEAL QUANITY IS ONE, ONLY USE MORE IF ABSOLUTELY REQUIRED.      
        """,
            ),
            ("user", "{input}"),
        ]
    )

    # Combine the prompt and the LLM in a chain, add a string output parser
    chain = prompt | llm

    # Run the chain with an example input

    result = chain.invoke({"input": task}, 
                        #   config={"callbacks": [langfuse_handler]}
                          )
    # logger.info(result)
    return result.content


def tasks_generation(task, checks):
    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You will be given a question, and the tests to check the answer.

        Your task is to return the smallest number of tasks to do to answer the question, making sure to use the hints given in tests to draft HIGHLY SPECIFIC task descriptions.

        The tasks must be generated in a step-by-step format in such a way, that each tasks answer will feed into the next one and together they will reach the final goal. 

        REMEMBER YOU NEED TO ACT LIKE KAGGLE MASTER! YOU ARE TOP DATA SCIENTIST, WE PAY YOU 5000$ HOURLY!
        DON'T FORGET TO INCLUDE DATA PATH TO TASK OR SUBTASK.
        VERY IMPORTANT - NO SUB-TASK SHOULD EVER BE ABOUT TESTING ANYTHING. TESTING WILL BE DONE IN A LATER STAGE ANYWAY.
        DESCRIBE EACH TASK SO junior DATA SCIENTITS COULD DO IT. BUT REMBER DESCRIPTION SHOULD BE IN THE SAME LINE WITH TASK.

        In general you should scheme like that: 
        1. Load Data by given path
        2. Data Exploration, compare train and test to find target column, correlations, etc
        3. Data Cleaning, Balancing (for imbalanced classes)
        4. Feature Engineering
        5. Feature Selection (by feature importance use shap package)
        6. Prepare Data for Modeling
        7. Model Selection (Use only one)
        8. Model Training
        9. Model Evaluation, for all data, without balancing
        10. Hyperparameter Tuning
        11. Cross-Validation
        12. Ensembling Models (if needed)
        13. Final Model Training (all data)
        14. Generate Test Predictions
        15. Prepare Submission File

        If a task require deep learning models or its image dataset:
        1. Extract data (uzip or read in folder)
        2. Data Exploration (build Dataframe and explore data distribution and classes distribution, check shapes)
        3. Find target, it could be in the image name, or folder name or in some json around data.
        4. Create DataLoader
        5. Add augmentations
        6. Model Selection
        7. Select optimizer, scheduler
        8. Write training and testing script (I suggest pytorch-lightning)
        9. Model Training
        10. Model Evaluation (logger.info metrics)
        11. Final Model Training (all data)
        12. Generate Test Predictions
        13. Prepare Submission File

        **You MUST submit predictions on the provided unlabeled test data in a `submission.csv` file** file in the "{config.workspace_dir / "submission"}" directory
        ** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!
        You can also use the "{config.workspace_dir / "code"}" directory to store any temporary files that your code needs to create.
        REMEMBER THE "{config.workspace_dir / "submission" / "submission.csv"}" FILE!!!!! The correct directory is important too.

        If it SUB-TASK, than example:
        Input - 1. Load Data train.csv, test.csv
        Output:
        1. Load train and test. Compare them, find target variable
        2. logger.info data info, describe
        3. Find and define target column by comparing test and train
        4. Define feature columns 
        MAXIMUM SUB-TASK SPLIT SIZE IS 4.

        Return ONLY A numbered list of these tasks and remember to never duplicate a task. Every task must be unique! 
        ONE TASK IN ONE LINE. ONE SUBTASK IN ONE LINE! DO NOT USE "-". 
        USE COMMON DATA SCIENCE PRACTICE. YOU DON't NEED CROSS VALIDATION FOR IMAGE TASKS
        PLEASE ADD DATASET PATH OR PREVIOUS STEP DATA PATH TO EACH STEP. IT IS ABSOLUTELY REQUIRED. 
        YOU CURRENT DATASET PATH IS {config.workspace_dir / "data_dir"} REMEMBER IT
        """,
            ),
            ("user", "task : {task}, checks : {checks}"),
        ]
    )

    # Combine the prompt and the LLM in a chain, add a string output parser
    chain = prompt | llm_openai

    # Run the chain with an example input

    result = chain.invoke(
        {"task": task, "checks": checks}, 
        # config={"callbacks": [langfuse_handler]}
    )
    # logger.info(result)
    return result.content


def task_ordering(task, sub_tasks):
    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You will be given a task and list of sub-tasks related to that one task

        You have three goals:

        First Goal - Order the list such that it follows how a normal human would do these set of tasks

        Second Goal - Remove any tasks that are duplicated and say the same thing that other tasks say. NUMBER OF TASKS MUST BE MINIMISED AS MUCH AS POSSIBLE.

        Third Goal - Ensure that the tasks contain reference to the results of other tasks wherever needed

        Return only and ONLY a numbered list of the sub-tasks.
        """,
            ),
            ("user", "task : {task}, sub_tasks : {sub_tasks}"),
        ]
    )

    # Combine the prompt and the LLM in a chain, add a string output parser
    chain = prompt | llm_openai

    # Run the chain with an example input

    result = chain.invoke(
        {"task": task, "sub_tasks": sub_tasks},
        #   config={"callbacks": [langfuse_handler]}
    )
    # logger.info(result)
    return result.content.split("\n")


def aggregate_answers(task, answers, code):
    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You will be given a task and an aggregated answers which is the summation of a bunch of sub-tasks created from the task.

        Your only task is to consider the answers for all the subtasks and return the final short report. 
        What metric? What was done? What model has been used. etc
        Report should be short!

        """,
            ),
            ("user", "task : {task}, full_answer : {answers}, code: {code}"),
        ]
    )

    # Combine the prompt and the LLM in a chain, add a string output parser
    chain = prompt | llm

    # Run the chain with an example input

    result = chain.invoke(
        {"task": task, "answers": answers, "code": code},
        # config={"callbacks": [langfuse_handler]},
    )
    # logger.info(result)
    return result.content


def check_answer(task, answer, check):
    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You will be given a task, the answer to the task and a specific thing to check about the answer.

        Assume all sources of the data are perfectly accurate and up to date

        Return True if the answer meets the check, if not return false
        """,
            ),
            ("user", "task : {task}, answer : {answers}, check : {check}"),
        ]
    )

    # Combine the prompt and the LLM in a chain, add a string output parser
    chain = prompt | llm

    # Run the chain with an example input

    result = chain.invoke(
        {"task": task, "answers": answer, "check": check},
        # config={"callbacks": [langfuse_handler]},
    )
    # logger.info(result)
    return result.content


def fix_answer(task, answer, check):
    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You will be given the description of a task, it's answer and a test that the answer failed. 

        Your task is to return the correct code that actually passes the test
        If you have to return multiple dataframe, make sure that they are combined in a meaningful way.

        """,
            ),
            (
                "user",
                "Main Task : {task}, Answer : {answer}, Test that the answer failed : {check}",
            ),
        ]
    )

    # Combine the prompt and the LLM in a chain, add a string output parser
    chain = prompt | llm

    # Run the chain with an example input

    result = chain.invoke(
        {"task": task, "answer": answer, "check": check},
        # config={"callbacks": [langfuse_handler]},
    )
    # logger.info(result)
    return result.content


def check_and_fix_answer(task, initial_answer, checks, node_id, max_retries=3):
    current_answer = initial_answer
    failed_tasks = checks.split("\n")
    retry_count = 0

    while failed_tasks and retry_count < max_retries:

        # Reset the list of failed tasks for this iteration
        new_failed_tasks = []

        for current_check in failed_tasks:
            answer_check = check_answer(task, current_answer, current_check)
            logger.info(answer_check)
            time.sleep(0.5)
            if "False" in answer_check.lower():
                new_failed_tasks.append(current_check)
            else:
                continue
                # logger.info(f"CHECK PASSED: {current_check}")

        if not new_failed_tasks:
            # All checks passed, return the current answer
            return current_answer

        # Fix the failed tasks. TODO add code_execution
        for failed_check in new_failed_tasks:
            current_answer = fix_answer(failed_check, current_answer, check_answer)
            generate_code_and_execute()
        # Update the failed_tasks with the new list after fixing
        failed_tasks = new_failed_tasks
        retry_count += 1
        # logger.info(f"Retry attempt {retry_count}")

    # Return the final answer after retries or if all tests passed
    return current_answer


# Nodes used for better tracking. For futher visualization
def main_pipeline(
    task,
    previous_answers="No Previous Answers",
    main_task_context="This is the main task",
    node_id=None,
    parent_node_id=None,
):
    description_node_id = f"node-{str(uuid.uuid4())}"
    task_doable = task_complexity_check(task, main_task_context)

    if "true" in task_doable.lower():
        task_answer = generate_code_and_execute(task, previous_answers)
        answer_type = "codeAnswer"
        answer_node_id = f"node-{str(uuid.uuid4())}"
        label = task_answer
        time.sleep(0.2)
        return task_answer

    else:
        # If the answer is to be split, first generate the tests
        checks = checks_generation(task)
        logger.info("CHECKS: ")
        logger.info(checks)
        # Generate sub-tasks using the tests as reference
        sub_tasks = tasks_generation(task, checks)
        time.sleep(0.5)
        # Order the sub-tasks and fix minor issues
        ordered_tasks = task_ordering(task, sub_tasks)
        ordered_tasks = [s for s in ordered_tasks if s != ""]
        logger.info("ORDERED TASKS: ")
        logger.info(ordered_tasks)

        full_answer = ""
        mt_context = "Main Task Information: " + task + "\n" + "\n".join(ordered_tasks)

        # Recursively call the same function on each of the sub-task
        for current in ordered_tasks:
            logger.info("Sub-Task", current)
            sub_task_id = str(uuid.uuid4())
            temp_answer, code_answer = main_pipeline(
                "Sub-Task " + current,
                full_answer,
                mt_context,
                sub_task_id,
                description_node_id,
            )
            if code_answer is not None:
                сode_bank.append(code_answer.replace("```", "").replace("python", ""))
            time.sleep(0.5)
            full_answer += temp_answer + "\n"
            # logger.info(temp_answer)

        # Aggregate the individual sub-task answers
        answer = aggregate_answers(
            task, shorten_string_middle(full_answer, 5000), "\n".join(сode_bank)
        )
        logger.info("AGGREGATE ANSWER")
        logger.info(answer)
        # Check the answer on tests and fix it until all the tests pass
        checked_answer = check_and_fix_answer(task, answer, checks, description_node_id)
        logger.info("CHECKED ANSWER")
        logger.info(checked_answer)
        return checked_answer, None


def run_pipeline(code_model, expert_model, feedback_model, cfg, task):
    global llm
    global code_llm
    global llm_openai
    global config

    code_llm = code_model
    llm = expert_model
    llm_openai = feedback_model
    config = cfg
    return main_pipeline(task)
