### C:\Users\tobia\Projekt\CreativeLLMs\.venv\Scripts\python.exe C:\Users\tobia\Projekt\CreativeLLMs\langchain-long-doc.py
### Note
`Number of requested results 7 is greater than number of elements in index 6, updating n_results = 6`

This occurs when you have a limited number of documents in Chroma. By default, the retriever uses similarity_search,
which has a default value of k=4. To address this, you can try adjusting the retriever's parameters
`qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))`

### Result
Question: Who is the CV about?
Response:  The CV is about Rachel Green, a graduate student at the University of Illinois at Urbana-Champaign.

Question: At what university did she study?
Response:  Rachel Green studied at the University of Illinois at Urbana-Champaign for her PhD and MA in English, and at Butler University in Indianapolis for her BA in English and Communications.

Question: What is the Dissertation title?
Response:  "Down on the Farm: World War One and the Emergence of Literary Modernism in the American South."

Question: Give examples of some Honors and Awards?
Response:  Some examples of honors and awards listed in this CV include the Jacob K. Javitz Fellowship, Graduate College Dissertation Completion Award, Campus Teaching Award, Doctoral Fellowship, Summer Research Grant, and Academic Scholarship.

Question: What is the capital of France?
Response:  Paris.

Process finished with exit code 0