import streamlit as st
import pandas as pd
#import plotly.figure.factory as ff
import plotly.express as px
st.set_page_config(layout='wide',page_title='My App')


html = '''
<html>
    <head>
        <title>This HTML App</title>
    </head>
    <body>
        <h1>This Long Text!!!</h1>
        <br>
        <hr>
        <h3>This a small text</h3>
    </body>
</html>
'''

with open('./com_html.html','r',encoding='utf-8') as f:
    filehtml = f.read()
    f.close()


# import matplotit pyplot as plt

# global variable
url = "https://www.youtube.com/watch?v=GdtCxmLCSEs"

# data app
df = pd.read_csv('./data/data.csv')

st.title('This is my first webapp!')
col1, col2 = st.columns((4,1))

with col1:
    with st.expander('Content1...'):
        st.subheader('Content1...')
        st.video(url)

    with st.expander('Content2_images..'):
        st.subheader('Content2_images..')
        st.image('./images/catdog.jpg')
        st.write('<h1>This is new title</h1>',unsafe_allow_html=True)
        st.markdown(html,unsafe_allow_html=True)

    with st.expander('Content2...'):
        st.subheader('Content2...')
        st.table(df)

    with st.expander('Content3_htmlContent..'):
        st.subheader('Content3_htmlContent...')
        import streamlit.components.v1 as htmlviewer
        htmlviewer.html(filehtml, height=800)



    # 총점 컬럼을 추가
    df['total'] = df[['kor', 'math', 'eng', 'info']].sum(axis=1)

    # 그래프 생성
    fig = px.bar(df, x='name', y='total', title='학생별 총점', text='total', color='name')
    st.plotly_chart(fig)

with col2:
    with st.expander('Tips...'):
        st.subheader('Tips...')