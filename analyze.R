# Author: Nick Sebasco
# Date: 3/20/2021
# Version: 2.0
# 
# Description: Use Knn to classify eye color as blue or brown.  Use three parameters: r, g, b.
# Part 1) classify input image of iris as blue or brown.
# Part 2) show k people with most similar iris as you.
# Part 3) Hypothesis: average euclidean distance is higher among the irises of blue eyed individuals.
# Part 4) 3D - colored plot of data.
# Part 5) Toggle k value and show the effect, attempt to find a pseudo-optimal k.
# part 6) (Future project) given a picture of a face, auto detect the eye and take a color sample of the iris.
library(imager)
library(dplyr)
library(purrr)
library(tictoc)

# 1) Set working directory
#---------------------------------
my_path <- "" # set path to KNN-IRIS-COLOR folder
if (getwd() != my_path) { # put working directory path here
  setwd(my_path)
  print(c("curr dir: ", getwd()))
} 


# 0) globals
#---------------------------------
data <- list()
blue_path <- 'data/blue_cropped_sample/'
blue_names <- list.files(path = "./data/blue_cropped_sample")
brown_path <- 'data/brown_cropped_sample/'
brown_names <- list.files(path = "./data/brown_cropped_sample")
test_vectors <- list()
k <- 5
useMedian <- FALSE
part1 <- FALSE # choose which experiments to run
part2 <- FALSE
part3 <- FALSE
part4 <- FALSE
part5 <- FALSE
part5_B <- TRUE

# 2) functions
#---------------------------------
img2rgb <- function (img, l, img_name, useMedian = FALSE) {
  # map an image to its central rgb value
  # useMedian <- FALSE, the mean will be used as central measure
  bdf <- as.data.frame(img)
  bdf <- mutate(bdf,channel=factor(cc,labels=c('R','G','B')))
  
  if (useMedian) {
    r_i <- median(bdf$value[bdf$channel=='R'])
    g_i <- median(bdf$value[bdf$channel=='G'])
    b_i <- median(bdf$value[bdf$channel=='B'])
  } else {
    r_i <- mean(bdf$value[bdf$channel=='R'])
    g_i <- mean(bdf$value[bdf$channel=='G'])
    b_i <- mean(bdf$value[bdf$channel=='B'])
  }
  list(rgb=c(r_i, g_i, b_i), label=l, img_name=img_name, raw=img)
}
euclidean.distance.custom <- function(v1, v2) {
  # custom implementation of euclidean distance between two double vectors v1 & v2
  t <- 0
  for (i in 1: length(v1)) {
    t <- t + (v1[i] - v2[i])^2 
  }
  t^0.5
}
euclidean.distance <- function(v1, v2) {
  # implementation of euclidean distance between v1 & v2 using the dist R function
  dist(rbind(v1, v2))
}
knn <- function(test_vector, data, k, dist_func) {
  # strategy 1) compute distances, sort, pick k-smallest distances
  # strategy 2) compute distances and maintain k-smallest distances in some collection type simultaneously. [not implemented]
  
  # strategy 1)
  data_w_distances <- list()
  i <- 1
  for (train_vector in data) {
    data_w_distances[[i]] <- list(rgb=train_vector$rgb, label=train_vector$label, dist=dist_func(train_vector$rgb, test_vector$rgb), img_name=train_vector$img_name)
    i <- i + 1
  }
  ordered <- data_w_distances[order(sapply(data_w_distances,'[[',3))]
  head(ordered, k) # choose k train vectors with the smallest distance between the test vector.
}

vote <- function(neighbors, showVotes = FALSE) {
  # neighbors <- list(list(rgb, label, dist), ...)
  # uses mode to find most frequent label, which will be the predicted label
  mapped <- map(neighbors, "label")
  counts = list()
  for (key in mapped) {
    if (exists(key, where=counts)) {
      counts[[key]] <- counts[[key]] + 1
    } else {
      counts[[key]] <- 1
    }
  }
  max.name <- ""
  max.val <- 0
  for (name in names(counts)) {
    if (counts[[name]] > max.val) {
      max.name <- name
      max.val <- counts[[name]]
    }
  }
  if (showVotes) {
    print(counts)
  }
  max.name
}

evaluate.model <- function(knn, tvs, data, k, dist_func) {
  # calculate classification error rate in testing data.
  count <- 0
  for (tv in tvs) {
    nearestNeighbors <- knn(tv, data, k, dist_func)
    if (tv$label == vote(nearestNeighbors)) {
      count <- count + 1
    }
  }
  count/ length(tvs)
}

find_k_images <- function(neighbors) {
  imgs <- list()
  i <- 1
  for (neighbor in neighbors) {
    fpath <- if(neighbor$label=="BROWN")'data/brown_people_sample/' else 'data/blue_people_sample/'
    fname <- gsub("_(l|r)","",neighbor$img_name)
    print(c("IMG NAME: ", paste0(fpath, fname)))
    imgs[[i]] <- load.image(paste0(fpath, fname))
    i <- i + 1
  }
  imgs
}

find_other_iris <- function(tv, useMedian = FALSE) {
  # given a test vector (image of a left or right iris) find the other iris
  fpath <- if(tv$label=="BROWN")'data/brown_cropped_sample/' else 'data/blue_cropped_sample/'
  fname <- if (grepl("_l", tv$img_name)) gsub("_l","_r",tv$img_name) else gsub("_r","_l",tv$img_name)
  #print(c("IMG NAME: ", paste0(fpath, fname)))
  img2rgb(load.image(paste0(fpath, fname)), tv$label, fname, useMedian)
}

build_color_vector <- function(data) {
  # transform rgb color into valid hex R color
  vector <- character()
  i <- 1
  for (tv in data) {
    vector[i] <- rgb(tv$rgb[1]*255, tv$rgb[2]*255, tv$rgb[3]*255, maxColorValue = 255) 
    i <- i + 1
  }
  vector
}

# 1) listing all blue/brown images and creating paths
#--------------------------------------------------------
for (i in  1:length(blue_names)) {
  blue_name_0 = blue_names[i]
  blue_0 = paste0(blue_path, blue_name_0)
  im_blue <- load.image(blue_0)
  data[[i]] <- img2rgb(im_blue, "BLUE", blue_name_0, useMedian = useMedian)
}
dat_len <- length(data) # needs data length after adding blue images.
for (i in  1:length(brown_names)) {
  brown_0 = paste0(brown_path, brown_names[i])
  im_brown <- load.image(brown_0)
  data[[i+dat_len]] <- img2rgb(im_brown, "BROWN", brown_names[i], useMedian = useMedian)
}

# Randomly select test vectors
#-----------------------------------------------------
test_vectors_ind <- sample(1:length(data), 3)
for (i in 1:length(test_vectors_ind)) {
  test_vectors[[i]] <- data[[test_vectors_ind[i]]]
  data[[test_vectors_ind[i]]] <- NULL
}


if (part1) {
  # Part 1.
  # get model accuracy on m length test set.
  acc = evaluate.model(knn, test_vectors, data, k, euclidean.distance.custom)
  print(paste0(sprintf("Accuracy on test %d vectors: ",length(test_vectors)),acc*100,"%"))
  
}



if (part2) {
  # Part 2
  # Read custom test vectors (me + girlfriends eye)
  # 1. see if we can correctly classify iris color
  # 2. find k - people with most similar iris colors
  test_path <- 'data/real_tests/'
  kk_test = paste0(test_path, "kk_cropped.png")
  im_kk <- load.image(kk_test)
  tv_kk <- img2rgb(im_kk, "BROWN", "kk_cropped.png")
  nearestNeighbors <- knn(tv_kk, data, k, euclidean.distance.custom)
  print(c("KK TEST: ", tv_kk))
  print("KK Neighbors: ")
  print(nearestNeighbors)
  print(c("PREDICTED (KK): ",vote(nearestNeighbors), " ACTUAL (KK): ",tv_kk$label))
  
  
  nick_test = paste0(test_path, "nick_cropped_2.png")
  im_nick <- load.image(nick_test)
  tv_nick <- img2rgb(im_nick, "GREEN", "nick_cropped_2.png", useMedian = TRUE)
  nearestNeighbors_nick <- knn(tv_nick, data, 3, euclidean.distance.custom)
  # print(c("NICK TEST: ", tv_nick))
  # print(c("PREDICTED (NICK): ",vote(nearestNeighbors_nick), " ACTUAL (NICK): ",tv_nick$label)) 
  
  imgs <- find_k_images(nearestNeighbors_nick)
  par(mfrow=c(k,1))
  #plot(im_kk) 
  #plot(im_nick)  
  for (img in imgs){
    plot(img)
  }
  
  print(
    "finished."
  )
}

if (part3) {
  # Part 3.
  # Find other iris given an iris.
  # quantifying iris dissimilarity between blue & brown eye people.
  # showing average euclidean distance among brown irises is less than blue irises =.
  
  tv <- test_vectors[[1]]
  nearestNeighbors <- knn(tv, data, k, euclidean.distance.custom)
  
  print(c("TEST: ", tv, tv$img_name))
  print("Neighbors:")
  print(nearestNeighbors)
  print(c("PREDICTED: ",vote(nearestNeighbors), " ACTUAL: ",tv$label))
  
  otherIris <- find_other_iris(tv,useMedian)
  print(c("Initial Iris: ", tv$img_name, "Other Iris: ", otherIris$img_name))
  
  brown_tot <- 0
  blue_tot <- 0
  blue_count <- 0
  brown_count <- 0
  
  for (tv1 in data) {
    if (grepl("_l",tv1$img_name)) {
      tv2 <- find_other_iris(tv1, useMedian)
      if (tv1$label == "BROWN") {
        brown_tot <- brown_tot + euclidean.distance(tv1$rgb, tv2$rgb)
        brown_count <- brown_count + 1
      } else {
        blue_tot <- blue_tot + euclidean.distance(tv1$rgb, tv2$rgb)
        blue_count <- blue_count + 1
      }
    }
  }
  
  print(c("Average brown distance: ", brown_tot/brown_count))
  print(c("Average blue distance: ", blue_tot/blue_count))
  
}

if (part4) {
  # Part 4.
  # Plot color space
  if (TRUE) { # PLOT color space
    library("scatterplot3d")
    plot_3d <- FALSE # IF true print plot of all 3 color dimensions, else only use 2.
    colors <- build_color_vector(data)
    plot_data <- list(x=double(), y=double(), z=double())
    i <- 1
    for (tv in data) {
      plot_data$x[i] <- tv$rgb[1]
      plot_data$y[i] <- tv$rgb[2]
      plot_data$z[i] <- tv$rgb[3]
      i <- i + 1
    }
    # 3D separation
    if (plot_3d) {
      scatterplot3d(x=plot_data$x, y=plot_data$y, z=plot_data$z,
                    main="RGB Eye Color Scatter Plot",
                    xlab = "red",
                    ylab = "green",
                    zlab = "blue",
                    color=colors,
                    pch=19,
                    grid=TRUE,
                    box=FALSE, cex.symbols=2)  
    } else {
      # set one of these variables to generate a particular plot.
      rb <- TRUE # red & blue
      rg <- TRUE # red & green
      bg <- TRUE # blue & green
      par(mfrow=c(1,as.numeric(rb) + as.numeric(rg) + as.numeric(bg)))
      if (rb) {
        #red & blue
        plot(x=plot_data$x, y=plot_data$z,
             main="RGB Eye Color Scatter Plot",
             xlab = "red",
             ylab = "blue",
             col=colors,
             pch=19,
             grid=TRUE,
             box=FALSE, cex.symbols=4, cex=3)
        #mtext("BLUE", side = 4, line = 3, cex = par("cex.lab"))
      } 
      if (rg) {
        #red & green
        plot(x=plot_data$x, y=plot_data$y,
             main="RGB Eye Color Scatter Plot",
             xlab = "red",
             ylab = "green",
             col=colors,
             pch=19,
             grid=TRUE,
             box=FALSE, cex.symbols=4, cex=3)
        #mtext("BLUE", side = 4, line = 3, cex = par("cex.lab"))
      }
      if (bg) {
        #blue & green
        plot(x=plot_data$y, y=plot_data$z,
             main="RGB Eye Color Scatter Plot",
             xlab = "green",
             ylab = "blue",
             col=colors,
             pch=19,
             grid=TRUE,
             box=FALSE, cex.symbols=4, cex=3)
        mtext("BLUE", side = 4, line = 3, cex = par("cex.lab"))
      }
    }
  }
}

if (part5) {
  # Part 5)
  # Evaluating k
  # 1) try every k-value from 1 to the length of the data set
  # 2) Plot error rate for each k-value 
  # 3) Find boundary points
  
  gen_master_data_frame <- FALSE
  gen_error_data_frame <- TRUE
  find_boundary_points <- FALSE
  
  if (gen_master_data_frame) {
    # attempt to classify each data point by using all other data points as the train set.  Try every possible
    # k-value.
    k_df <- list(name=character(), actual_label=character(), predicted_label=character(), k=integer())
    idx <- 1
    for (ki in 1:length(data)) {
      for (i in 1:length(data)) {
        data_cpy <- data
        tv <- data[[i]]
        data_cpy[[i]] <- NULL
        nearestNeighbors <- knn(tv, data_cpy, ki, euclidean.distance.custom)
        predicted <- vote(nearestNeighbors)
        k_df[["name"]][idx] <- tv$img_name
        k_df[["actual_label"]][idx] <- tv$label
        k_df[["predicted_label"]][idx] <- predicted
        k_df[["k"]][idx] <- ki + 0
        idx <- idx + 1
      }
    }
    k_df <- as.data.frame(k_df)
  }
  
  if (gen_error_data_frame) {
    # attempt to classify each data point by using all other data points as the train set.  Try every possible
    # k-value.
    kerr_df <- list(k=integer(), error=character())
    idx <- 1
    for (ki in 1:length(data)) {
      correct <- 0
      for (i in 1:length(data)) {
        data_cpy <- data
        tv <- data[[i]]
        data_cpy[[i]] <- NULL
        nearestNeighbors <- knn(tv, data_cpy, ki, euclidean.distance.custom)
        predicted <- vote(nearestNeighbors)
        if (predicted == tv$label) {
          correct <- correct + 1
        }
      }
      kerr_df[["error"]][idx] <- paste0(sprintf("%.2f",100* (length(data) - correct) / length(data)),"%")
      kerr_df[["k"]][idx] <- ki
      idx <- idx + 1
    }
    kerr_df <- as.data.frame(kerr_df)
    plot(
      kerr_df$k, 
      map(kerr_df$err,function(x){ as.numeric(gsub("%","",x))}),
      main="K vs. Classification Error Rate",
      xlab="k",
      ylab="Error (%)",
      col="purple",
      type="l"
      )
    mtext("Error (%)", side = 4, line = 3, cex = par("cex.lab"))
  }
  if (find_boundary_points) {
    # Choose k = 25, or (66) after looking at plot: "k v. classification error"
    # assume all misclassified points lie on the boundary.
    boundary <- list()
    idx <- 1
    ki <- 25 # 66
    for (i in 1:length(data)) {

      data_cpy <- data
      tv <- data[[i]]
      data_cpy[[i]] <- NULL
      nearestNeighbors <- knn(tv, data_cpy, ki, euclidean.distance.custom)
      predicted <- vote(nearestNeighbors)
      
      if (predicted != tv$label) { # found a boundary point
        boundary[[idx]] <- tv
        idx <- idx + 1
      }
      
    }
    print(c("length boundary: ", length(boundary)))
  }
}
# Part 5. continued 
if (part5_B) {
  # time different k-values
  #   k0    k1    k2   k3   k4    k5
  #    2     5    10   25   50   100
  # 1.11  1.07  1.17 1.46 1.87  2.70
  k <- 7
  trials <- 500
  tic(sprintf("Timing %d trials @ k = %d",trials,k))
  for (i in 1: trials) {
    acc = evaluate.model(knn, test_vectors, data, k, euclidean.distance.custom)
  }
  print("set laeo.")
  toc()
}

# print(c(r_i, g_i, b_i))
# par(mfrow=c(2,1)) # using par to combine multiple plots
# plot(im_blue) #plot blue eye
# plot(im_brown) #plot blue eye
