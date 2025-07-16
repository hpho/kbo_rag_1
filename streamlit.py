import streamlit as st
import streamlit.components.v1 as components
import random
import uuid

# 세션 상태 초기화
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = [{"role": "assistant", "content": "May I help you?"}]
    
if "use_external_docs" not in st.session_state:
    st.session_state.use_external_docs = False
    # st.session_state.use_cont_msg = False
if "example_questions" not in st.session_state:
    st.session_state.example_questions = [
        "23년 홈런왕 찾아줘.",
        "24년 다승왕 찾아줘.",
        "23년 김현수 선수의 안타개수는?",
    ]
if "random_questions" not in st.session_state:
    st.session_state.random_questions = random.sample(st.session_state.example_questions, 3)

# 메시지 추가 함수
def add_message(role, content):
    chat_id = st.session_state.current_chat
    st.session_state.chats[chat_id].append({"role": role, "content": content})

# 채팅 전환 함수
def switch_chat(chat_id):
    st.session_state.current_chat = chat_id
    st.rerun()
    
# figure 추가 함수
def add_fig(role, content, dir_):
    chat_id = st.session_state.current_chat
    st.session_state.chats[chat_id].append({"role":role, "content": content, "figure": dir_})

user_id = "HKW"
depart_info = "admin"

with st.sidebar:
    st.markdown(f"**이름 : {user_id}**")
    st.markdown(f"**소속 : {depart_info}**")
    st.divider()
    
    for chat_id in st.session_state.chats:
        if len(st.session_state.chats[chat_id]) > 1:
            if st.sidebar.button(f"""{st.session_state.chats[chat_id][1]['content']}""", key=chat_id):
                switch_chat(chat_id)
                
    if st.sidebar.button("📝 New Chat"):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chats[new_chat_id] = [{"role": "assistant", "content": "May I help you?"}]
        switch_chat(new_chat_id)
        

st.title("💬 KBO STATIZ AI 검색")
st.caption("🚀 **하강우**")

st.subheader("💡 Example Questions")
for i in range(1):
    cols = st.columns(3)  # 가로로 3개 열 만들기
    for idx, col in enumerate(cols):
        question = st.session_state.random_questions[i*3 + idx]
        if col.button(question, use_container_width=True):  # key를 다르게 해야 함
            send_question_to_queue(question)
        
# 사용자 입력 처리
if user_input := st.chat_input():
    add_message("assistant",f"{user_input}에 대한 답은 다음과 같습니다. 이것은 echo bot 입니다.")
    
# 채팅 메시지 출력
# st.subheader("Conversation")
st.divider()
for msg in st.session_state.chats[st.session_state.current_chat]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"], unsafe_allow_html=True)
        
        if msg.get("figure"):
            st.image(msg["figure"])
