#Retrieve the data saved AFTER the profiling practice...... this means data already cleaned


dd <- read.csv("./Datasets/pokemonProcessed.csv", header=T, sep = ",");
names(dd)
dim(dd)
summary(dd)

attach(dd)

#set a list of numerical variables
names(dd)

dcon <- data.frame (Total,Sp_Atk,Sp_Def,Speed,Attack,Defense,HP, Catch_Rate)
dim(dcon)

#
# CLUSTERING
#



# KMEANS RUN, BUT HOW MANY CLASSES?

k1 <- kmeans(dcon,5)
names(dcon)
print(k1)

attributes(k1)

k1$size

k1$withinss

k1$centers

# LETS COMPUTE THE DECOMPOSITION OF INERTIA

Bss <- sum(rowSums(k1$centers^2)*k1$size)
Bss
Wss <- sum(k1$withinss)
Wss
Tss <- k1$totss
Tss

Bss+Wss

Ib1 <- 100*Bss/(Bss+Wss)
Ib1

# LETS REPEAT THE KMEANS RUN WITH K=5

k2 <- kmeans(dcon,5)
k2$size

Bss <- sum(rowSums(k2$centers^2)*k2$size)
Bss
Wss <- sum(k2$withinss)
Wss

Ib2 <- 100*Bss/(Bss+Wss)
Ib2
Ib1

k2$centers
k1$centers

plot(k1$centers[,3],k1$centers[,2])

table(k1$cluster, k2$cluster)

# WHY WE HAVE OBTAINED DIFFERENT RESULTS?, AND WHICH RUN IS BETTER?

# NOW TRY K=8

k3 <- kmeans(dcon,8)
k3$size

Bss <- sum(rowSums(k3$centers^2)*k3$size)
Wss <- sum(k3$withinss)

Ib3 <- 100*Bss/(Bss+Wss)
Ib3


# HIERARCHICAL CLUSTERING

d  <- dist(dcon)
h1 <- hclust(d,method="ward.D")  # NOTICE THE COST
plot(h1)

d  <- dist(dcon)
h1 <- hclust(d,method="ward")  # NOTICE THE COST
plot(h1)

# BUT WE ONLY NEED WHERE THERE ARE THE LEAPS OF THE HEIGHT

# WHERE ARE THER THE LEAPS? WHERE WILL YOU CUT THE DENDREOGRAM?, HOW MANY CLASSES WILL YOU OBTAIN?

nc = 4

c1 <- cutree(h1,nc)

c1[1:20]

nc = 5

c5 <- cutree(h1,nc)

c5[1:20]


table(c1)
table(c5)
table(c1,c5)


cdg <- aggregate(as.data.frame(dcon),list(c1),mean)
cdg

plot(cdg[,1], cdg[,7])

# LETS SEE THE PARTITION VISUALLY


plot(Total, Catch_Rate,col=c1,main="Clustering of credit data in 4 classes")
legend("topright",c("class1","class2","class3", "class4"),pch=1,col=c(1:4))





pairs(dcon[,1:7], col=c1)

#plot(FI[,1],FI[,2],col=c1,main="Clustering of credit data in 3 classes")
#legend("topleft",c("c1","c2","c3"),pch=1,col=c(1:3))

# LETS SEE THE QUALITY OF THE HIERARCHICAL PARTITION



Bss <- sum(rowSums(cdg^2)*as.numeric(table(c1)))

Ib4 <- 100*Bss/Tss
Ib4


#move to Gower mixed distance to deal 
#simoultaneously with numerical and qualitative data

library(cluster)

#dissimilarity matrix

actives<-c(2:16)
dissimMatrix <- daisy(dd[,actives], metric = "gower", stand=TRUE)

distMatrix<-dissimMatrix^2

h1 <- hclust(distMatrix,method="ward.D")  # NOTICE THE COST
#versions noves "ward.D" i abans de plot: par(mar=rep(2,4)) si se quejara de los margenes del plot

plot(h1)

c2 <- cutree(h1,4)

#class sizes 
table(c2)

#comparing with other partitions
table(c1,c2)


names(dd)
#ratiFin
boxplot(Total~c2, horizontal=TRUE)

#plazo
boxplot(Catch_Rate~c2, horizontal=TRUE)

#gastos
boxplot(dd[,9]~c2, horizontal=TRUE)

names <- c("isLegendary","Total", "Catch_Rate")
pairs(dd[,names], col=c2)

cdg <- aggregate(as.data.frame(dcon),list(c2),mean)
cdg

#Profiling plots
