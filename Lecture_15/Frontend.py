
import streamlit as st
from Backend import chatbot
from Backend import title_llm
from Backend import retrive_all_threads, save_chat_title, retrieve_all_chat_titles
from langchain_core.messages import HumanMessage
import uuid

# ********************************* Utility Functions ******************************************

def genereate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = genereate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']

def generrate_title(user_input):

    prompt = f"""
Generate a short tile (3-6 words) forr a chatbot conversation
based on the user input message.

Return only the title

User Message: {user_input}
"""
    response = title_llm.invoke(prompt)
    return response.content


# ********************************** Session Setup ********************************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = genereate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_all_threads()

if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = retrieve_all_chat_titles()

add_thread(st.session_state['thread_id'])

# **************************************** SIDEBAR UI ******************************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversation')

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(st.session_state['chat_titles'].get(thread_id, "No Title"), key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': message.content})

        st.session_state['message_history'] = temp_messages



# **************************************** MAIN Ui ********************************************

# Display previous messages
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.write(message['content'])

user_input = st.chat_input('Type here')


config = {
    'configurable': {'thread_id': st.session_state['thread_id']},
    "metadata": {
        "thread_id": st.session_state['thread_id']
    },
    "run_name": "chat_turn",
    
    }

if user_input:

    if st.session_state['thread_id'] not in st.session_state['chat_titles']:
        title = generrate_title(user_input)
        st.session_state['chat_titles'][st.session_state['thread_id']] = title

        thread = str(st.session_state['thread_id'])
        save_chat_title(thread_id, title)
        title_generated = True
    else:
        title_generated = False


    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    with st.chat_message('user'):
        st.write(user_input)

    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            try:
                # res = chatbot.invoke(
                #     {'messages': [HumanMessage(content=user_input)]},
                #     config=config
                # )
                res = ""
                response_placeholder = st.empty()
                for message_chunk, metadata in chatbot.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode='messages'
                ):
                    if message_chunk.content:
                        res += message_chunk.content
                        response_placeholder.markdown(res)

                assistant_output = res

                st.session_state['message_history'].append({
                    'role': 'assistant',
                    'content': assistant_output
                })

            except Exception as e:
                st.error(f'Error: {e}')

    if title_generated:
        st.rerun()
