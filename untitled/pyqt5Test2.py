from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

b = QPushButton(root, text="Enter", width=10, height=2, command=button1)
c = QPushButton(root, text="Clear", width=10, height=2, command=clear)
b.grid(row=0,column=0, sticky=W)
c.grid(row=0,column=1, sticky=W)

textframe = Frame(root)
textframe.grid(in_=root, row=1, column=0, columnspan=3, sticky=NSEW)
root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

text = Text(root, width=35, height=15)
scrollbar = Scrollbar(root)
scrollbar.config(command=text.yview)
text.config(yscrollcommand=scrollbar.set)
scrollbar.pack(in_=textframe, side=RIGHT, fill=Y)
text.pack(in_=textframe, side=LEFT, fill=BOTH, expand=True)