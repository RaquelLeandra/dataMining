# Regression
# Karina Gibert
# karina.gibert@upc.es
# Abril 2017


dd <- read.table("./Datasets/pokemonProcessed.csv", header=T, sep = ",")

attach(dd)
names(dd)

#plot(x,y)
plot(Total,Catch_Rate) 



# linear regression of y function of x 
reg1 <- lm(Total ~ Catch_Rate, data=dd) 
print (reg1) 
summary(reg1)
abline(b =-1.057, a =523.939,col = 'red' )

class(reg1)
# structure of object produced 
attributes(reg1) 
# access to part of the results 
reg1$coefficients 
reg1$fitted.values

plot(reg1)


# getting details of a command
help(lm)

# Check the fit visually 
plot(Total,Catch_Rate) 
# and without closing the plot window 
lines(Catch_Rate, reg1$fitted.values, col="red")


plot(Total, Catch_Rate)
reg2 <- lm(Total ~ Catch_Rate, data=dd) 
print (reg2) 
summary(reg2)

plot(reg2)


# Check the fit visually 
plot(Total, Catch_Rate)
# and without closing the plot window 
lines(Total, reg2$fitted.values, col="red")

reg3 <- lm(Catch_Rate ~ Total+Species, data=dd) 
print (reg3) 
summary(reg3)

plot(reg3)

attributes(reg3)
reg3$xlevels
reg3$terms

# Check the fit visually 
plot(Total, Catch_Rate, col=Species)
# and without closing the plot window 
lines(Total, reg3$fitted.values, col="blue")



setwd("D:/karina/docencia/areferenciesPPT/0DadesPractiques/CREDSCO/")
dd <- read.table("credscoClean.csv",header=T, sep=";")
names(dd)
sapply(dd, class)

attach(dd)

plot(Importe.solicitado, Estalvi) 

# linear regression of y function of x 
reg4 <- lm(Estalvi ~ Importe.solicitado, data=dd) 
print (reg4) 
summary(reg4)

plot(reg4)

# Check the fit visually 
plot(Importe.solicitado, Estalvi) 
# and without closing the plot window 
lines(Importe.solicitado, reg4$fitted.values, col="red")


plot(Importe.solicitado, Estalvi, col=Tipo.trabajo) 

reg5 <- lm(Estalvi ~ Importe.solicitado+Tipo.trabajo, data=dd) 
summary(reg5)
summary(reg4)

plot(reg5)

# Check the fit visually 
plot(Importe.solicitado, Estalvi, col=Tipo.trabajo) 
# and without closing the plot window 
lines(Importe.solicitado, reg4$fitted.values, col="black")

Y=log(Estalvi)
plot(Importe.solicitado, Y) 

#aproximacio classica. No es massa bona idea
# linear regression of y function of x 
reg6 <- lm(Y ~ Importe.solicitado, data=dd) 

dades<-data.frame(Y, Importe.solicitado)

dades<-na.omit(dades)
dim(dades)
dim(dd)
summary(dades)
summary(Y)

dades[is.infinite(dades[,1]),1]<-NA
dades<-na.omit(dades)
dim(dades)
dim(dd)
summary(dades)
summary(Y)


reg6 <- lm(dades[,1] ~ dades[,2], data=dd,) 
summary(reg6)

plot(reg6)


# Check the fit visually 
plot(dades[,2], dades[,1]) 
# and without closing the plot window 
lines(dades[,2], reg6$fitted.values, col="red")


#cas extrem
x <- rnorm(100) 
?rnorm

#rpois(n, lambda) Llei de poisson
# a look at the content of x 
x

# basic descriptive statistics: Numerical variables
summary(x) 

#sort values of x
x<-sort(x)

# generate a second variable 
y <- sin(x*pi)+0.1*rnorm(100)  
y  

# basic descriptive statistics for y 
summary(y) 
hist(y) 

# and its relationship with x 
plot(x,y) 

# linear regression of y function of x 
reg7 <- lm(y ~ x, data=dd) 
summary(reg7)

plot(reg7)

plot(x,y) 
# and without closing the plot window 
lines(x, reg7$fitted.values, col="red")

# Change regression algorithm by a local regression
reg8 <- loess(y ~ x, data=dd) 
print (reg8) 
attributes(reg8)


# and plot the new fit in the same display
lines(x,reg8$fitted,col= "blue")

reg9 <- loess(y ~ poly(x,10), data=dd) 
plot(x,y) 
lines(x,reg9$fitted,col= "blue")

# perform a polinomic regression 
# introducing powers of x: 
reg10 <- lm(y ~ poly(x,10), data=dd) 
lines(x,reg10$fitted.values,col= "red") 

reg11 <- lm(y ~ poly(x,6), data=dd) 
lines(x,reg11$fitted.values,col= "green") 

#what happens iwith bigger degrees?
reg20 <- lm(y ~ poly(x,20), data=dd)
lines(x,reg20$fitted.values,col= "cyan")  

#remember parsimonious principle

# Use this model to predict new observations. 

#Simulate new observations following the model plus some noise
xnew <- rnorm(100, mean=0.3)
xnew <- sort(xnew)
x
ynew <- sin(xnew*pi*0.9)+0.2*rnorm(100) 


# Inspect new dataset 
plot(xnew,ynew) 


# Use the chosen model to predict this new data 
ypred1 <- predict(reg10,new.data=xnew) 
lines(xnew,ypred1) 
ypred2 <- predict(reg7,new.data=xnew) 
lines(xnew,ypred2, col="red") 

#Some objective way to assess the quality of the fit 
rss <- sum((ynew-ypred1)^2) 
rss 
# Who got the best model? 
# is this measure rss reliable?
