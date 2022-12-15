#install.packages("RamanMP")
library(RamanMP)

# Data
data("MPdatabase")
data("matrix_unknown")

plot(matrix_unknown$freq, matrix_unknown$unknown_1, type="l")

# Normalization
matrix_unknown_norm <- norm.min.max(matrix_unknown)
plot(matrix_unknown_norm$freq, matrix_unknown_norm$unknown_1, type="l")

matrix_unknown_norm_snv <- norm.SNV(matrix_unknown)
plot(matrix_unknown_norm_snv$freq, 
     matrix_unknown_norm_snv$unknown_1, type="l")

# find peaks
peak.finder(spectrum, threshold=0, m=100, max.peak=0)

# remove regios
region.remove(MPdatabase[,c(1,6)], min.region=500, max.region=1200) 

# smooth (needs to remove NA first)
smooth.vect <- savit.gol(matrix_unknown[1:2046,3], 
                         filt=3, 
                         filt_order=10,
                         der_order=0)
plot(matrix_unknown[1:2046,3], type='l')
lines(smooth.vect, type='l', col='red')


# aligne spectra
x <- spectra.alignment(norm.min.max(MPdatabase[,c(1:10)]),
                       norm.min.max(matrix_unknown), 
                       t=0)

plot(x$freq, x$unknown_3, type = 'l')
plot(x$freq, x$ABS_1, type = 'l')
lines(x$freq, ifelse(is.na(x$unknown_3),0, x$unknown_3))

# spectra matching

data("MPdatabase","single_unknown")
identif_spectra<-spectra.corr(MPdatabase, single_unknown, t=0.5, normal='min.max')

# matching process too slow 


# slopp and sloppe libraries

data_slopp <- read.csv('~/Documents/Research_Abuj/spectral_libraries/df_slopp_inter.csv')
data_slopp_e <- read.csv('~/Documents/Research_Abuj/spectral_libraries/df_slopp_e_inter.csv')
data_slopp_pivot <- read.csv('~/Documents/Research_Abuj/spectral_libraries/df_slopp_inter_pivot.csv')
data_slopp_pivot_norm <-norm.SNV(data_slopp_pivot)

data_slopp_norm <- melt(data_slopp_pivot_norm, id="freq")
data_slopp$file_name_2 = gsub(" ", ".", data_slopp$file_name)
data_slopp_norm <- data_slopp_norm %>% select(c(wavelength=freq, intensity_norm=value, file_name_2=variable))
data_slopp_norm = left_join(data_slopp, data_slopp_norm, by=c("file_name_2", "wavelength"))

data_slopp_norm <- data_slopp_norm %>% 
  select(-c(file_name_2, intensity)) %>%
  select(wavelength, intensity=intensity_norm, polymer, file_name, color)

data_slopp_pivot <- data_slopp_pivot %>% select(-c(X))
spect_names_slopp_pivot <- colnames(data_slopp_pivot)
data_slopp_pivot <- data_slopp_pivot %>% select(freq=wavelength, c(spect_names_slopp_pivot[2:length(spect_names_slopp_pivot)]))

plot(norm.min.max(MPdatabase)[,1], norm.min.max(MPdatabase)[,10], type = 'l')
lines(norm.min.max(data_slopp_pivot)[,1], norm.min.max(data_slopp_pivot)[,2], type = 'l')

poly_name_white <- unique(data_slopp %>% filter(color == "White") %>% select(file_name))
data_slopp_pivot_white <- data_slopp_pivot[,c('freq', gsub(" ", ".", unlist(poly_name_white$file_name)))]

identif_spectra<-spectra.corr(data_slopp_pivot, single_unknown, t=0.5, normal='min.max')
identif_spectra<-spectra.corr(MPdatabase, single_unknown, t=0.5, normal='min.max')
identif_spectra<-spectra.corr(MPdatabase, data_slopp_pivot[,c('freq', 'Polyethylene.6..White.Microsphere.txt')], t=0.5, normal='min.max')
#identif_spectra<-spectra.corr(MPdatabase, data_slopp_pivot[,c('freq', gsub(" ", ".", unlist(poly_name_white$file_name)))], t=0.5, normal='min.max')

identif_spectra<-spectra.corr(MPdatabase, data_slopp_pivot_white[, c(1,2)], t=0.5, normal='min.max')
identif_spectra<-spectra.corr(MPdatabase, data_slopp_pivot_white[, c(1,3)], t=0.5, normal='min.max')
identif_spectra<-spectra.corr(MPdatabase, data_slopp_pivot_white[, c(1,3)], t=0.5, normal='min.max')

data_slopp_pivot_w <- dcast(data_slopp, "file_name + polymer +color ~ wavelength", value.var="intensity")
data_slopp_pivot_w$color <- as.factor(data_slopp_pivot_w$color)
data_slopp_pivot_w$polymer <- as.factor(data_slopp_pivot_w$polymer)
colnames(data_slopp_pivot_w)[4:length(colnames(data_slopp_pivot_w))] = paste0('w_', as.integer(colnames(data_slopp_pivot_w)[4:length(colnames(data_slopp_pivot_w))]))

data_slopp_norm_pivot_w <- dcast(data_slopp_norm, "file_name + polymer +color ~ wavelength", value.var="intensity")
data_slopp_norm_pivot_w$color <- as.factor(data_slopp_norm_pivot_w$color)
data_slopp_norm_pivot_w$polymer <- as.factor(data_slopp_norm_pivot_w$polymer)
colnames(data_slopp_norm_pivot_w)[4:length(colnames(data_slopp_norm_pivot_w))] = paste0('w_', as.integer(colnames(data_slopp_norm_pivot_w)[4:length(colnames(data_slopp_norm_pivot_w))]))


pc <- prcomp(data_slopp_pivot_w[,-c(1,2,3,4,length(data_slopp_pivot_w))],
             center = TRUE,
             scale. = TRUE)
#attributes(pc)
#summary(pc)

dim(t(pc$rotation))
pc_data = pc$x

plot(pc_data[,1], pc_data[,2])

error_vector = c()

for (npc in 2:20) {
  error_vector_aux = c()
  for (i in 1:100) {
    #npc = 5
    train = sample(148, size = 148 *.7, replace = F)
    y_train <- as.character(data_slopp_pivot_w$polymer[train])
    data_train <- data.frame(polymer = y_train, pc_data[train,1:npc])
    
    y_test <- data_slopp_pivot_w$polymer[-train]
    x_test <- pc_data[-train,1:npc]
    res.multnom <- multinom(polymer ~ ., data=data_train)
    
    prediction_test <- predict(res.multnom, x_test)
    accuracy_vector_aux[i] = mean(as.character(prediction_test) == as.character(y_test))
  }
  error_vector[npc] = mean(error_vector_aux)
}


train = sample(148, size = 148 *.7, replace = F)
y_train <- as.character(data_slopp_pivot_w$polymer[train])
data_train <- data.frame(polymer = y_train, pc_data[train,1:npc])


y_test <- data_slopp_pivot_w$polymer[-train]
x_test <- pc_data[-train,1:npc]
res.multnom <- multinom(polymer ~ ., data=data_train)

prediction_test <- predict(res.multnom, x_test)
mean(as.character(prediction_test) == as.character(y_test))

plot(error_vector)


cast(data_slopp_pivot)

x <- data_slopp_norm_pivot_w[,-c(1,2,3,4,length(data_slopp_norm_pivot_w))]
#x <- data_slopp_pivot_w[,-c(1,2,3,4,length(data_slopp_pivot_w))]
#x <- x[-134,]

slopp_polymer_list <- unique(as.character(data_slopp_norm_pivot_w$polymer))
slopp_polymer_list = slopp_polymer_list[slopp_polymer_list != "Polymethyl Methacrylate"]

slopp_polymer_dict = data.frame(class=1:(length(slopp_polymer_list)), 
                                polymer=slopp_polymer_list)

x <- x[as.character(data_slopp_norm_pivot_w$polymer) %in% slopp_polymer_dict$polymer,]

y <- merge(data.frame(polymer=data_slopp_norm_pivot_w$polymer), 
                 slopp_polymer_dict, by="polymer")$class
#y <- y[-134]

x_Polyethylene <- x[y == 8,]
plot(unlist(x_Polyethylene[1,]))

x_Polyethylene_test <- x_mat_tets[y_test == "Polyethylene",]

plot(unlist(x_Polyethylene[1,]))
lines(unlist(x_Polyethylene_test[1,]))

train = sample(length(y), size = length(y) *0.7, replace = F)
x_train = x[train,]
y_train = y[train]

x = x[-133,]
y = y[-133]

cvfit <- cv.glmnet(x=as.matrix(x), y=y, family = "multinomial", type.multinomial = "grouped")
plot(cvfit)

fit <- glmnet(x_train, y_train, family = "multinomial", type.multinomial = "grouped")
plot(fit, xvar = "lambda", label = TRUE, type.coef = "2norm")

x_mat_tets <- x[-train,]
y_test = y[-train]

y_pred = predict(fit, newx = as.matrix(x[-train,]), s = 0.000001, type = "class")[,1]
mean(y_test == y_pred)
y_pred = predict(fit, newx = as.matrix(x[-train,]), s = 0.00001, type = "class")[,1]
mean(y_test == y_pred)
y_pred = predict(cvfit, newx = as.matrix(x[-train,]), s = "lambda.min", type = "class")
mean(y_test == y_pred)

table(as.numeric(y_pred), y_test)


MPdatabase_inter_norm_pivot_w <- dcast(MPdatabase_inter_norm, "file_name + polymer ~ wavelength", value.var="intensity")
MPdatabase_inter_norm_pivot_w$polymer <- as.factor(MPdatabase_inter_norm_pivot_w$polymer)
colnames(MPdatabase_inter_norm_pivot_w)[4:length(colnames(MPdatabase_inter_norm_pivot_w))] = paste0('w_', as.integer(colnames(MPdatabase_inter_norm_pivot_w)[4:length(colnames(MPdatabase_inter_norm_pivot_w))]))

x_mat_tets <- as.matrix(MPdatabase_inter_norm_pivot_w[!is.na(MPdatabase_inter_norm_pivot_w$polymer),-c(1,2,3,1402)])
y_test <- as.character(MPdatabase_inter_norm_pivot_w[!is.na(MPdatabase_inter_norm_pivot_w$polymer), 2])

y_test_2 = left_join(data.frame(polymer=as.character(y_test)), 
      slopp_polymer_dict, by="polymer")$class

y_pred = predict(cvfit, newx = x_mat_tets, s = "lambda.min", type = "class")
mean(y_test == y_pred)

mean(y_test_2 == y_pred, na.rm=T)


table(y_test, y_pred)
