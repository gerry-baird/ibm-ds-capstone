# Inferring Disposable Income from Property Values and Supermarket Proximity
Assests from IBM Data Science Capstone Project

# Introduction
If you’d ever been shopping for groceries in the UK you wouldn’t need a statistical genius to tell you which of the two towns illustrated below probably had the highest disposable income. We don’t need to know anything about the household earnings to form an opinion based on the average property price and the presence of certain supermarkets. 

![](https://github.com/gerry-baird/ibm-ds-capstone/blob/master/img/towns.jpg)

Beyond this intuitive understanding is it possible to quantify this somehow by applying machine learning? Can we leverage the detailed analysis performed by the supermarket chains themselves to infer anything about disposable income? 

Grocery shopping is dominated by large supermarket chains, but they target different customer segments. Premium supermarkets are operated by Waitrose, mainstream supermarkets are operated by Tesco, Sainsburys, Asda and Morrisons. Budget supermarkets are operated by Lidl and Aldi. Can we leverage the detailed analysis performed by these supermarket chains that led to them to open stores in some areas and not others to infer anything about these areas in terms of disposable income? 

The presence of a supermarket can raise property prices, so the two features being analysed aren’t truly independent. A study by Lloyds bank in 2017 suggested that a Waitrose can add as much as £36k to local property prices. £22k for a mainstream supermarket and £6k for a budget supermarket [1].

# Data
I used property transaction data published by the UK Govt Land Registry on Kaggle, combined with geolocation data from Google, augmented with location insights provided by FourSquare. 

## Property Transaction Data
The transaction data is hosted on Kaggle [2]. There is 2.2 GB of data in the complete dataset, containing property transactions from 1995 until 2017. An example of the data is shown below.

![](https://github.com/gerry-baird/ibm-ds-capstone/blob/master/img/data.jpg)

For this project we are interested in the price, the transaction date, the town, district and county. One significant limitation of the data is that precise location data isn’t provided, we have to infer the location from town, district in county. This is rather imprecise for large cities resulting in a lack of detail for any large city such as Birmingham, London or Liverpool as shown above.

## Geolocation Data
Google provides a geolocation API as part of the Google Maps platform [3]. Given an address, the geolocation API will try to provide a precise latitude and longitude even if the address is ambiguous or incomplete. The example response below shows the location information returned by the API for Google’s head office.

![](https://github.com/gerry-baird/ibm-ds-capstone/blob/master/img/google.jpg)

## Venue Data
The FourSquare Places API provides details of venues within a radius of a given location [4]. This can be a general search or a search within a specific category of venue such as a supermarket. For the purposes of this exercise I was searching for supermarket venues within a 3000 meter radius of a given longitude and latitude.   

The example below shows the search result for Morpeth in Northumberland. 

![](https://github.com/gerry-baird/ibm-ds-capstone/blob/master/img/venue.jpg)

For this search there were 3 venues found within the search radius where the category was supermarket. I’ve expanded the second result which is for a Morrisons supermarket. 



