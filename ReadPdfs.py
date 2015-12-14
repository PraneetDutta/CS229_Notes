from bs4 import BeautifulSoup
from subprocess import call

def makePDFS(section,prefix,loc):
    links = section.find_all("a")
    for a in links:
        href = a["href"]
        url = prefix + href
        names = href.split("/")
        name = names[len(names)-1]
        call(["wget", url])
        call(["mv",name,loc+name])

f = open("./index.html","r")
file_text = f.read()
bs_text = BeautifulSoup(file_text,'html.parser')
f.close()
sections = bs_text.findAll('ul', {'type': 'disc'})
#0 = homeworks
#1 = lectures
#2 = sections
prefix = "http://cs229.stanford.edu/"
loc = ["./homeworks/","./lectures/","./sections/"]
for i in range(len(sections)):
    section = sections[i]
    makePDFS(section,prefix,loc[i])
