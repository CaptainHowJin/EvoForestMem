import json
import re
from pathlib import Path

PROMPTS = json.loads((Path(__file__).resolve().parents[1]/'config/adaptive_answer_protocol.json').read_text())
JUDGE_SCHEMA = {'type':'object','properties':{'label':{'type':'string','enum':['CORRECT','WRONG']}},
                'required':['label'],'additionalProperties':False}


def question_type(row):

    return 'multiple_choice' if str(row['answer']).strip() in list('ABCD') else 'open_ended'


def answer_messages(row, context):
    kind = question_type(row)
    return format_answer_messages(row['question'], context, kind, row.get('options'))


def answer_format(row):

    kind = question_type(row)
    return {'question_type': kind, 'options': row.get('options') if kind == 'multiple_choice' else None}


def format_answer_messages(question, context, question_type='open_ended', options=None):
    kind = question_type
    if kind == 'multiple_choice':
        options = options or {}
        if set(options) != set('ABCD'):
            raise ValueError('MC options are required')
        question = '\n'.join([question.rstrip(), '', 'Options:',
            *[f'{letter}. {str(options[letter]).strip()}' for letter in sorted(options)],
            '', 'Select the correct option by its letter (A/B/C/D).'])
    prompt = PROMPTS['answer'][kind].format(context=context.strip() or '(No results)',question=question,options='')
    return [{'role':'user','content':prompt}]


def parse_mc(response):
    if response.startswith('[LLM_CALL_FAILED]'):
        return '[FAILED]'
    response = response.strip().upper()
    if not response:
        return '[EMPTY]'
    if len(response)==1 and response in 'ABCD':
        return response
    for pattern in (r'\b([ABCD])[.):,\s]', r'(?:answer|choice|option|select)[:\s]+([ABCD])\b'):
        match = re.search(pattern,response,re.I)
        if match:
            return match.group(1).upper()
    if response[0] in 'ABCD' and (len(response)==1 or not response[1].isalpha()):
        return response[0]
    if response[-1] in 'ABCD' and (len(response)==1 or not response[-2].isalpha()):
        return response[-1]
    return '[INVALID]'


def judge_messages(row, answer):
    prompts = PROMPTS['llm_judge']
    return [{'role':'system','content':prompts['system_prompt']},
            {'role':'user','content':prompts['user_prompt'].format(question=row['question'],
                golden_answer=row['answer'],generated_answer=answer)}]
