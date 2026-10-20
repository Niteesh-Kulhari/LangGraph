
import streamlit as st
from chatbot import chatbot
from langchain_core.messages import HumanMessage

config = {'configurable': {'thread_id': '1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Display previous messages
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.write(message['content'])

user_input = st.chat_input('Type here')

if user_input:
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
