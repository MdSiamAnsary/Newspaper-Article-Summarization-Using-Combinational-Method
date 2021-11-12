import tkinter as tk
from tkinter import ttk
import tkinter as tk
import datetime
from tkinter import font
from tkinter import scrolledtext
from tkinter import *
from tkinter import messagebox
from DBHandler import *
from datetime import datetime
from newspaper import Article
import numpy as np
import pandas as pd
from rake_nltk import *
from nltk.tokenize import sent_tokenize
import nltk
from textblob import TextBlob
# nltk.download('stopwords')
from nltk.corpus import stopwords
# nltk.download('punkt') # one time execution
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import warnings
warnings.simplefilter("ignore")

# ------ ------ DB object and connection variables
db = "summarizationDBFour.db"
conn = sqlite3.connect(db)
cur = conn.cursor()


class Application(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Newspaper Article Summarizer")

        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)


        self.frames = {}

        for F in ( PageOne, PageTwo):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(PageOne)

    def show_frame(self, targetFrame):
        frame = self.frames[targetFrame]
        frame.tkraise()

















class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='white')

        tk.Frame.option_add(self, "*font", "Cambria 13")

        self.OPTIONS = [
            "Business",
            "Entertainment",
            "Politics",
            "Sports",
            "Tech"
        ]

        self.article = StringVar()
        self.generated_summ = StringVar()
        self.url = StringVar()
        self.user = StringVar()

        self.q1 = StringVar()
        self.q2 = StringVar()

        self.wrapper1 = LabelFrame(self, text="", bg='white')
        self.wrapper2 = LabelFrame(self, text="Provide Inputs", bg='white')
        self.wrapper3 = LabelFrame(self, text="Fetched Newspaper Article", bg='white')
        self.wrapper4 = LabelFrame(self, text="", bg='white')
        self.wrapper5 = LabelFrame(self, text="Generated Summary", bg='white')

        self.wrapper2.pack(fill="both", padx=20, pady=6)
        self.wrapper3.pack(fill="both", padx=20, pady=6)
        self.wrapper4.pack(fill="both", padx=20, pady=6)
        self.wrapper5.pack(fill="both", padx=20, pady=6)
        self.wrapper1.pack(fill="both", padx=20, pady=6)

        self.lbl_uname = Label(self.wrapper2, text="Enter Username:", bg='white')
        self.lbl_uname.grid(row=0, column=0, padx=10, pady=5, sticky=W)
        self.ent_uname = Entry(self.wrapper2, textvariable=self.q1, bg='peach puff', width=40)
        self.ent_uname.grid(row=0, column=1, columnspan=2, padx=6, sticky=W)

        self.lbl_url = Label(self.wrapper2, text="Enter URL:", bg='white')
        self.lbl_url.grid(row=1, column=0, padx=10, pady=5, sticky=W)
        self.ent_url = Entry(self.wrapper2, textvariable=self.q2, bg='peach puff', width=80)
        self.ent_url.grid(row=1, column=1, columnspan=4, padx=6, sticky=W)

        self.article_cat = StringVar()
        self.article_cat.set(self.OPTIONS[0])
        self.lbl_art_cat = Label(self.wrapper2, text="Select category:", bg='white')
        self.lbl_art_cat.grid(row=2, column=0, padx=5, pady=3, sticky=W)
        self.dropmenu = OptionMenu(self.wrapper2, self.article_cat, *self.OPTIONS)
        self.dropmenu.grid(row=2, column=1, padx=5, pady=3)

        self.btn = Button(self.wrapper2, text="Fetch Article", command= lambda:self.fetcharticle())
        self.btn.grid(row=3, column=1, padx=6, pady=5)
        self.cbtn = Button(self.wrapper2, text="Clear Fields", command=lambda:self.clearinputs())
        self.cbtn.grid(row=3, column=3, padx=6, pady=5)

        # self.lbl1 = Label(self.wrapper3, text="Article Text", bg='white')
        # self.lbl1.grid(row=0, column=0, padx=5, pady=3)
        self.ent1 = scrolledtext.ScrolledText(self.wrapper3, wrap=tk.WORD, height=8, width=95, bg='bisque2')
        self.ent1.grid(row=0, column=1, padx=5, pady=3)

        self.ent2 = scrolledtext.ScrolledText(self.wrapper5, wrap=tk.WORD, height=8, width = 95, bg='bisque2')
        self.ent2.grid(row=1, column=1, padx=5, pady=3)



        self.summbtn = Button(self.wrapper4, text="Summarize", width=40, command= lambda:self.summarize() , bg='light green', font=('Cambria', '13', 'bold'))
        self.summbtn.grid(row=1, column=10, padx=250, pady=5)

        self.svDB_btn = Button(self.wrapper1, text="Save in Database", command=lambda: self.savedindb(), bg='bisque3',
                               width=20)
        self.svTXT_btn = Button(self.wrapper1, text="Save in .txt File", command=lambda: self.saveintxt(), bg='bisque3',
                                width=20)
        self.clear_btn = Button(self.wrapper1, text="Clear Fields", command=lambda: self.clear(), bg='bisque3',
                                width=20)
        self.records_btn = Button(self.wrapper1, text="Check Records", command=lambda: controller.show_frame(PageTwo),
                                  bg='bisque3', width=20)

        self.svDB_btn.grid(row=3, column=3, padx=15, pady=10)
        self.svTXT_btn.grid(row=3, column=5, padx=15, pady=10)
        self.clear_btn.grid(row=3, column=7, padx=15, pady=10)
        self.records_btn.grid(row=3, column=9, padx=15, pady=10)


        #self.close_btn = Button(self.wrapper1, text="Back", command=lambda: controller.show_frame(StartPage) , bg='bisque1', width=20)


        #self.close_btn.grid(row=0, column=4, columnspan=3, padx=35, pady=10)

    def sentiment_score_neu(self, sen):
        if len(sen) > 0:
            if TextBlob(sen).sentiment.polarity == 0:
                return 1
            else:
                return 0
        else:
            return 0

    def sentiment_score_pos(self, sen):
        if len(sen) > 0:
            if TextBlob(sen).sentiment.polarity > 0:
                return 1
            else:
                return 0
        else:
            return 0

    def sentiment_score_neg(self, sen):
        if len(sen) > 0:
            if TextBlob(sen).sentiment.polarity < 0:
                return 1
            else:
                return 0
        else:
            return 0

    '''
    def sentimentScore(self, sen):
        if (len(sen) > 0):
            if (TextBlob(sen).sentiment.polarity == 0):
                return 1
            else:
                return 0
        else:
            return 0
    '''

    # function to remove stopwords
    def remove_stopwords(self, sen):
        sen_new = " ".join([i for i in sen if i not in self.stop_words])
        return sen_new


    def summarize(self):

        if self.ent_url.get() == '' or self.ent_uname.get() == '':
            messagebox.showerror('Required Field', 'Please enter appropriate value in input field')
            return
        else:

            self.url = self.ent_url.get()
            self.user = self.ent_uname.get()

            article_text = ''
            try:
                article = Article(self.url)
                article.download()
                article.parse()
                article_text = article.text
            except:
                article_text = ''



            self.article = article_text

            if self.article == '' :
                messagebox.showerror('Required Field', 'Please input a proper URL')
                return
            else:

                document_sentiment = TextBlob(self.article).sentiment.polarity

                split_text = self.article.split("\n")
                non_empty_lines = [line for line in split_text if line.strip() != ""]
                file_text = ""
                for line in non_empty_lines:
                    file_text += line + "\n"
                with open('fetchedtext.txt', 'w') as f:
                    f.write(file_text)

                with open('fetchedtext.txt') as f:
                    lines = f.readlines()

                total_sen_count = len(lines)

                try:
                    article_category =  str(self.article_cat.get())
                except:
                    article_category =  "Entertainment"

                # article_category =  str(self.article_cat.get())

                if article_category == "Sports":
                    with open('sport_phraselist.pkl', 'rb') as f_s:
                        phrase_list = pickle.load(f_s)

                elif article_category == "Politics":
                    with open('politics_phraselist.pkl', 'rb') as f_p:
                        phrase_list = pickle.load(f_p)

                elif article_category == "Entertainment":
                    with open('entertainment_phraselist.pkl', 'rb') as f_e:
                        phrase_list = pickle.load(f_e)

                elif article_category == "Business":
                    with open('business_phraselist.pkl', 'rb') as f_b:
                        phrase_list = pickle.load(f_b)

                elif article_category == "Tech":
                    with open('tech_phraselist.pkl', 'rb') as f_t:
                        phrase_list = pickle.load(f_t)


                self.sentences = []
                for line in lines:
                    self.sentences.append(sent_tokenize(line))
                self.sentences = [y for x in self.sentences for y in x]

                # self.sentences = sent_tokenize(lines)

                print(self.sentences)





                self.sentiment_scores = []
                self.keyphrase_based_scores = []

                for each_sen in self.sentences:
                    if document_sentiment == 0:
                        self.sentiment_scores.append(self.sentiment_score_neu(each_sen))
                    elif document_sentiment > 0:
                        self.sentiment_scores.append(self.sentiment_score_pos(each_sen))
                    else:
                        self.sentiment_scores.append(self.sentiment_score_neg(each_sen))

                    r = Rake()
                    r.extract_keywords_from_text(each_sen)
                    phrases_in_each_sen = r.get_ranked_phrases()

                    phr_count = 0
                    for each_phr in phrases_in_each_sen:
                        if each_phr in phrase_list:
                            phr_count = phr_count + 1
                    if phr_count == 0:
                        self.keyphrase_based_scores.append(0)
                    else:
                        self.keyphrase_based_scores.append(phr_count / len(each_sen))

                    # self.sentiment_scores.append(self.sentimentScore(each_sen))

                word_embeddings = {}
                f = open('glove.6B.100d.txt', encoding='utf-8')
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    word_embeddings[word] = coefs
                f.close()

                # remove punctuations, numbers and special characters
                self.clean_sentences = pd.Series(self.sentences).str.replace("[^a-zA-Z]", " ")

                # make alphabets lowercase
                self.clean_sentences = [s.lower() for s in self.clean_sentences]

                self.stop_words = stopwords.words('english')

                # remove stopwords from the sentences
                self.clean_sentences = [self.remove_stopwords(r.split()) for r in self.clean_sentences]

                '''
                # Extract word vectors
                word_embeddings = {}
                f = open('glove.6B.100d.txt', encoding='utf-8')
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    word_embeddings[word] = coefs
                f.close()
                '''

                self.sentence_vectors = []
                for i in self.clean_sentences:
                    if len(i) != 0:
                        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (
                                    len(i.split()) + 0.001)
                    else:
                        v = np.zeros((100,))
                    self.sentence_vectors.append(v)

                # similarity matrix
                sim_mat = np.zeros([len(self.sentences), len(self.sentences)])

                for i in range(len(self.sentences)):
                    for j in range(len(self.sentences)):
                        if i != j:
                            sim_mat[i][j] = \
                                cosine_similarity(self.sentence_vectors[i].reshape(1, 100),
                                                  self.sentence_vectors[j].reshape(1, 100))[
                                    0, 0]

                import networkx as nx

                nx_graph = nx.from_numpy_array(sim_mat)
                scores = nx.pagerank(nx_graph)

                for i in range(int(total_sen_count)):
                    scores[i] = scores[i] + self.sentiment_scores[i] + self.keyphrase_based_scores[i]


                self.generated_summary_sentences = []
                ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(self.sentences)), reverse=True)
                # Extract top 40 percent sentences as the summary
                for i in range(int((total_sen_count * 40) / 100)):
                    # print(ranked_sentences[i][1] + "\n")
                    # self.summ = self.summ + " " + ranked_sentences[i][1]
                    self.generated_summary_sentences.append(ranked_sentences[i][1])

                self.summ = ""
                for eachSen in self.sentences:
                    if eachSen in self.generated_summary_sentences:
                        # summary = summary + str(TextBlob(sen).correct()) + " "
                        self.summ = self.summ + " " + str(TextBlob(eachSen).correct()) + " "

                # self.generated_summ.set(self.summ)

                self.ent2.configure(state='normal')
                self.ent2.delete('1.0', END)
                self.ent2.insert(INSERT, self.summ)
                self.ent2.configure(state='disabled')




    def fetcharticle(self):

        if self.ent_url.get() == '' or self.ent_uname.get() == '':
        #if self.ent_url.get() == '':
            messagebox.showerror('Required Field', 'Please enter appropriate value in input field')
            return
        else:
            self.ent1.configure(state='normal')
            self.ent1.delete('1.0', END)

            self.url = self.ent_url.get()
            self.user = self.ent_uname.get()

            article_text = ''
            try:
                article = Article(self.url)
                article.download()
                article.parse()
                article_text = article.text
            except:
                article_text = ''



            self.article = article_text

            #if self.article == '' or self.summary == '':
            if self.article == '' :

                messagebox.showerror('Required Field', 'Please input a proper URL')
                return
            else:

                self.ent1.insert(INSERT, self.article)
                #self.ent2.insert(INSERT, self.summary)
                self.ent1.configure(state='disabled')
                #self.ent2.configure(state='disabled')



    def clearinputs(self):
        self.ent_url.delete(0, END)

        self.ent1.configure(state='normal')
        self.ent1.delete('1.0', END)
        self.ent1.configure(state='disabled')





    def savedindb(self):


        if self.summ != '':

            cur.execute("INSERT INTO summarytable VALUES (NULL,?, ?, ?, ?)",
                        (self.user, self.url, self.article, self.summ))

            conn.commit()

            messagebox.showinfo("", "Information has been stored in database")
            self.clear()
        else:
            return True



    def saveintxt(self):

        if str(self.summ) == '':
            return
        else:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            dt_string = re.sub(r"[/:]", " ", dt_string)
            file_name = "Summarization file " + dt_string + ".txt"

            article = re.sub(r'\n+', '\n', self.article)
            summary = re.sub(r'\n+', '\n', self.summ)

            text_file = open(file_name, "w")
            text_file.write("Newspaper Article URL: " + self.url + "\n\n\n")
            text_file.write("Newspaper Article: \n\n\n" + article + "\n\n\n")
            text_file.write("Summary: \n\n\n" + summary + "\n")
            text_file.close()

            full_path = os.path.realpath(__file__)
            path, filename = os.path.split(full_path)
            path = os.path.realpath(path)
            os.startfile(path)

            

        

    def clear(self):

        self.ent1.configure(state='normal')
        self.ent2.configure(state='normal')
        self.ent1.delete('1.0', END)
        self.ent2.delete('1.0', END)
        self.generated_summ.set('')
        self.summary = ''
        self.article_cat.set(self.OPTIONS[0])
        # self.article_cat= self.OPTIONS[0]
        self.article = ''
        self.url = ''
        self.summ = ''

    def close(self):
        return True

    def checkrecords(self):
        return True



















class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='white')


        self.record_user = StringVar()
        self.record_url = StringVar()
        self.record_article = StringVar()
        self.record_summary = StringVar()


        self.qPT = StringVar()  # String variable for the username to search in records

        self.t1PT = StringVar()  # String variable to hold the summary id for deleting

        # Three portions in the window
        self.wrapper1PT = LabelFrame(self, text="Summary Records", bg='white')
        self.wrapper2PT = LabelFrame(self, text="Search by Username", bg='white')
        self.wrapper3PT = LabelFrame(self, text="Selected Record", bg='white')

        self.wrapper2PT.pack(fill="both", padx=20, pady=10)  # Input Searching
        self.wrapper1PT.pack(fill="both", padx=20, pady=10)  # Records
        self.wrapper3PT.pack(fill="both", padx=20, pady=10)  # Particular Record

        # ----- ----- Search for records by username
        self.lblPT = Label(self.wrapper2PT, text="Enter Username:", font=('Cambria', 12), bg='white')
        self.lblPT.pack(side=tk.LEFT, padx=10, pady=5)

        # Input for user whose records are to be displayed
        self.entPT = Entry(self.wrapper2PT, textvariable=self.qPT, font=('Cambria', 12), bg='peach puff', width=40)
        self.entPT.pack(side=tk.LEFT, padx=6)

        self.btnPT = Button(self.wrapper2PT, text="Search", command=lambda : self.searchPT(), font=('Cambria', 12))
        self.btnPT.pack(side=tk.LEFT, padx=6)

        self.cbtnPT = Button(self.wrapper2PT, text="Clear", command=lambda : self.clearPT(), font=('Cambria', 12))
        self.cbtnPT.pack(side=tk.LEFT, padx=6)
        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

        self.showall = Button(self.wrapper1PT, text="Display all records", command=lambda: self.clearPT(), font=('Cambria', 11))
        self.showall.pack(side=tk.LEFT, padx=6)

        # ----- ----- Treeview widget

        # Style for the Treeview widget
        self.stylePT = ttk.Style()
        self.stylePT.configure("mystyle.Treeview", highlightthickness=0, bd=0,
                        font=('Cambria', 11))  # Modify the font of the body
        self.stylePT.configure("mystyle.Treeview.Heading", font=('Cambria', 11))  # Modify the font of the headings
        # style.layout("mystyle.Treeview", [('mystyle.Treeview.treearea', {'sticky': 'nswe'})]) # Remove the borders

        # Five columns
        self.trvPT = ttk.Treeview(self.wrapper1PT, columns=(0, 1, 2, 3, 4),
                           show="headings",
                           height="5",
                           style="mystyle.Treeview")

        # Padding added
        self.trvPT.pack(pady=10)

        self.headersPT = ('Id', 'User', 'URL', 'Text', 'Summary')
        self.widthsPT = (5, 200, 700, 5, 5)
        # Setting column widths
        self.trvPT['show'] = 'headings'
        self.trvPT['columns'] = self.headersPT
        for i, header in enumerate(self.headersPT):
            self.trvPT.heading(header, text=header)
            self.trvPT.column(header, width=self.widthsPT[i])

        self.trvPT["displaycolumns"] = (1, 2)  # Only display User and URL



        self.clearPT()

        # Show all records by default
        cur.execute("SELECT * FROM summarytable")
        self.rowsPT = Database(db).fetch()  # Using the Database class of DBHandler
        self.updatePT(self.rowsPT)

        # For geting the text and summary of a particular record
        self.trvPT.bind('<Double 1>', self.getrowPT)  # Get a row by double click



        # ----------------------------------------------------------------
        # Of a particular selected record,
        # We display, user, url, text and summary

        # ----- ----- For displaying user
        self.lbl1_textVarPT = StringVar()
        self.lbl1_textVarPT.set("User : ")
        self.lbl1PT = Label(self.wrapper3PT, textvariable=self.lbl1_textVarPT, bg='white')
        self.lbl1PT.grid(row=0, column=0, columnspan=5, padx=5, pady=3, sticky=W)

        # ----- ----- For displaying URL
        self.lbl2_textVarPT = StringVar()
        self.lbl2_textVarPT.set("Article URL : ")
        self.lbl2PT = Label(self.wrapper3PT, textvariable=self.lbl2_textVarPT, bg='white')
        self.lbl2PT.grid(row=1, column=0, columnspan=5, padx=5, pady=3, sticky=W)

        # ----- ----- For displaying the article
        self.lbl3PT = Label(self.wrapper3PT, text="Article Text:", bg='white')
        self.lbl3PT.grid(row=2, column=0, padx=5, pady=3)

        self.ent3PT = scrolledtext.ScrolledText(self.wrapper3PT, wrap=tk.WORD, height=6, bg='bisque2', width=85)
        self.ent3PT.grid(row=2, column=1, columnspan=3, padx=5, pady=3)

        # ----- ----- For displaying the summary
        self.lbl4PT = Label(self.wrapper3PT, text="Summary:", bg='white')
        self.lbl4PT.grid(row=3, column=0, padx=5, pady=3)

        self.ent4PT = scrolledtext.ScrolledText(self.wrapper3PT, wrap=tk.WORD, height=5, bg='bisque2', width=85)
        self.ent4PT.grid(row=3, column=1, columnspan=3, padx=5, pady=3)
        # ---------------------------------------------------------------------------

        # What do we want to do of the selected record ? Save in .txt / Clear / Delete the record ?
        self.save_btnPT = Button(self.wrapper3PT, text="Save in .TXT", command=lambda : self.save_in_txtPT(), bg='bisque3', width=20)
        self.clear_btnPT = Button(self.wrapper3PT, text="Clear", command=lambda : self.clear_selectedPT(), bg='bisque3', width=20)
        self.delete_btnPT = Button(self.wrapper3PT, text="Delete Selected", command=lambda : self.delete_recordPT(), bg='bisque3', width=20)

        self.save_btnPT.grid(row=5, column=1, padx=5, pady=10)
        self.delete_btnPT.grid(row=5, column=2, padx=5, pady=10)
        self.clear_btnPT.grid(row=5, column=3, padx=5, pady=10)


        button2 = Button(self, text="Back",
                             command=lambda: controller.show_frame(PageOne),
                             bg= 'bisque3', width = 20,
                             font = ('Cambria', 12))
        button2.pack()

    # This method shows the records in treeview according to conditions
    def updatePT(self, rows):
        self.trvPT.delete(*self.trvPT.get_children())
        for i in rows:
            self.trvPT.insert('', 'end', values=i)

    # This method is used to display the records of a particular user
    def searchPT(self):
        q2 = self.qPT.get()
        query = "SELECT * FROM summarytable WHERE username LIKE'%" + q2 + "%'"
        cur.execute(query)
        rows = cur.fetchall()
        self.updatePT(rows)

    # This method clears the user input for looking records by user
    # and shows all records for all users
    def clearPT(self):
        self.qPT.set("")
        query = "SELECT * FROM summarytable"
        cur.execute(query)
        rows = cur.fetchall()
        self.updatePT(rows)

    # To show the text and summary of a particular record
    def getrowPT(self, event):

        rowid = self.trvPT.identify_row(event.y)
        item = self.trvPT.item(self.trvPT.focus())

        self.record_user = item['values'][1]
        self.record_url = item['values'][2]
        self.record_article = item['values'][3]
        self.record_summary = item['values'][4]

        self.t1PT.set(item['values'][0])  # holds the id, in case for deletion

        self.lbl1_textVarPT.set("User : " + item['values'][1])
        self.lbl2_textVarPT.set("Article URL : " + item['values'][2])
        self.ent3PT.configure(state='normal')
        self.ent4PT.configure(state='normal')
        self.ent3PT.delete('1.0', END)
        self.ent4PT.delete('1.0', END)
        self.ent3PT.insert(INSERT, item['values'][3])
        self.ent4PT.insert(INSERT, item['values'][4])
        self.ent3PT.configure(state='disabled')
        self.ent4PT.configure(state='disabled')

    # Method for deleting the records
    def delete_recordPT(self):
        if self.t1PT.get() == '':
            messagebox.showerror("Required Fields", "Nothing selected")
        else:
            summary_id = self.t1PT.get()
            if messagebox.askyesno("Confirm Delete?", "Delete the record"):

                query = "DELETE FROM summarytable WHERE id = " + summary_id
                cur.execute(query)
                conn.commit()
                self.clearPT()
            else:
                return True

            self.clear_selectedPT()

    def save_in_txtPT(self):

        try:

            if self.record_summary != '':

                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                dt_string = re.sub(r"[/:]", " ", dt_string)
                file_name = "Summarization file " + dt_string + ".txt"

                article = re.sub(r'\n+', '\n', self.record_article)
                summary = re.sub(r'\n+', '\n', self.record_summary)

                text_file = open(file_name, "w")
                text_file.write("Username: " + self.record_user + "\n\n\n")
                text_file.write("Newspaper Article URL: " + self.record_url + "\n\n\n")
                text_file.write("Newspaper Article: \n\n\n" + article + "\n\n\n")
                text_file.write("Summary: \n\n\n" + summary + "\n")
                text_file.close()

                full_path = os.path.realpath(__file__)
                path, filename = os.path.split(full_path)
                path = os.path.realpath(path)
                os.startfile(path)

            else:
                return
        except:
            pass

    # Clear out the selected record
    def clear_selectedPT(self):

        self.record_url = ''
        self.record_user = ''
        self.record_article = ''
        self.record_summary = ''

        self.t1PT.set('')
        self.lbl1_textVarPT.set("User : ")
        self.lbl2_textVarPT.set("Article URL : ")

        self.ent3PT.configure(state='normal')
        self.ent3PT.delete('1.0', END)
        self.ent3PT.configure(state='disabled')

        self.ent4PT.configure(state='normal')
        self.ent4PT.delete('1.0', END)
        self.ent4PT.configure(state='disabled')



app = Application()

# set window size
app.geometry("950x720")

# init menubar
menubar = tk.Menu(app)

# creating the menus
menuManage = tk.Menu(menubar, tearoff=0)



# menu: manage
menuManage.add_command(label="P1", command=lambda: app.show_frame(PageOne))
menuManage.add_command(label="P2", command=lambda: app.show_frame(PageTwo))


# attach menubar
app.config(menu=menubar, bg='white')
app.resizable(False, False)
app.mainloop()
