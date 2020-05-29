# Inferring Disposable Income from Property Values and Supermarket Proximity
Assests from IBM Data Science Capstone Project

# Introduction
If you’d ever been shopping for groceries in the UK you wouldn’t need a statistical genius to tell you which of the two towns illustrated below probably had the highest disposable income. We don’t need to know anything about the household earnings to form an opinion based on the average property price and the presence of certain supermarkets. 

![](https://github.com/gerry-baird/ibm-ds-capstone/blob/master/img/towns.jpg)

Beyond this intuitive understanding is it possible to quantify this somehow by applying machine learning? Can we leverage the detailed analysis performed by the supermarket chains themselves to infer anything about disposable income? 

Grocery shopping is dominated by large supermarket chains, but they target different customer segments. Premium supermarkets are operated by Waitrose, mainstream supermarkets are operated by Tesco, Sainsburys, Asda and Morrisons. Budget supermarkets are operated by Lidl and Aldi. Can we leverage the detailed analysis performed by these supermarket chains that led to them to open stores in some areas and not others to infer anything about these areas in terms of disposable income? 

The presence of a supermarket can raise property prices, so the two features being analysed aren’t truly independent. A study by Lloyds bank in 2017 suggested that a Waitrose can add as much as £36k to local property prices. £22k for a mainstream supermarket and £6k for a budget supermarket [1].

