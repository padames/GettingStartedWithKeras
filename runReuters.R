#!/usr/bin/env Rscript

## Example of a multi-class classification example from the Keras package
# Based on the example in section 3.5 of the book Deep Learning with R, by Francois Chollet.



list.of.packages <- c("keras")
# installed.packages() returns a matrix of characters
new.packages <- list.of.packages[!(list.of.packages %in% 
                                     installed.packages()[,"Package"])]
if(length(new.packages)) 
  install.packages(new.packages)

library(keras)

reuters <- dataset_reuters(num_words = 10000)

# Keras multiple assignment operator
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters

# each sample is a list of integers (word indices)

newswire.number <- readline(prompt = "What newswire # would you like to see?")
newswire.number <- as.numeric(newswire.number)
tryCatch(stopifnot(c(newswire.number > 0, newswire.number < 10000)), 
         finally = print(paste0("Choose from 1 to 10,000 ", geterrmessage())))

newswire <- train_data[[newswire.number]]
list.data.sample2 = paste(sapply(newswire, FUN = paste, collapse = " "), collapse = " ")
print(paste0("A sample is already a list of numbers: [", list.data.sample2, "]"))
rm(list.data.sample2)

# Decoding newswires back to text
word_index <- dataset_reuters_word_index()

reverse_word_index <- names(word_index)

names(reverse_word_index) <- word_index

decoded_newswire <- sapply(newswire, function(index) {
  word <- if(index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

list.data.sample2.words = paste(decoded_newswire, collapse = " ")
print(paste0("The same sample decoded: ", list.data.sample2.words))
rm(list.data.sample2.words)


# Preparing the data


# prepare to vectorize newswires:

vectorize_sequences <- function(sequences, dimension = 10000) {
  results = matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

print(paste0("Vectorizing input and lables.."))
      
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)


# preparing the labels using the Keras function

one_hot_train_labels <- to_categorical(train_labels)
one_hot_test_labels <- to_categorical(test_labels)


## Building the network

# The hidden layers have 64 units to attempt to separate 46 classes.
# The softwax of size 46 predicts a probability distribution over the 46 classes.


model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

print("Compiling network")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

## Setting aside a validation set

print("setting aside a validation set from the training set")

val_indices <- 1:1000

# first the train data
x_val <- x_train[val_indices,]
partial_x_trian <- x_train[-val_indices,]

#then the train labels
y_val <- one_hot_train_labels[val_indices,]
partial_y_train <- one_hot_train_labels[-val_indices,]

### Training the model

print("Training the model on 20 epochs")

history <- model %>% fit(
  partial_x_trian,
  partial_y_train,
  epochs = 20,
  batch_size = 512, # 2^9
  # batch_size = 256, # 2^8
  # batch_size = 64, # 2^6 this makes overfits worse and validation loss much higher
  validation_data = list(x_val, y_val)
)

plot(history)