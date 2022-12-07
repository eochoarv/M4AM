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

data_slopp <- read.csv('~/Documents/Research_Abuj/spectral_libraries/df_slopp_inter.csv')
data_slopp_pivot <- read.csv('~/Documents/Research_Abuj/spectral_libraries/df_slopp_inter_pivot.csv')

ploy_name_white <- unique(data_slopp %>% filter(color == "White") %>% select(file_name))

data_slopp_pivot <- data_slopp_pivot %>% select(-c(X))

spect_names_slopp_pivot <- colnames(data_slopp_pivot)

data_slopp_pivot <- data_slopp_pivot %>% select(freq=wavelength, c(spect_names_slopp_pivot[2:length(spect_names_slopp_pivot)]))

plot(norm.min.max(MPdatabase)[,1], norm.min.max(MPdatabase)[,10], type = 'l')
lines(norm.min.max(data_slopp_pivot)[,1], norm.min.max(data_slopp_pivot)[,2], type = 'l')


identif_spectra<-spectra.corr(data_slopp, single_unknown, t=0.5, normal='min.max')

identif_spectra<-spectra.corr(MPdatabase, single_unknown, t=0.5, normal='min.max')

identif_spectra<-spectra.corr(MPdatabase, data_slopp_pivot[,c('freq', 'Polyethylene.6..White.Microsphere.txt')], t=0.5, normal='min.max')

identif_spectra<-spectra.corr(MPdatabase, data_slopp_pivot[,c('freq', gsub(" ", ".", unlist(ploy_name_white$file_name)))], t=0.5, normal='min.max')

data_slopp_pivot_white <- data_slopp_pivot[,c('freq', gsub(" ", ".", unlist(ploy_name_white$file_name)))]

identif_spectra<-spectra.corr(MPdatabase, data_slopp_pivot_white[, c(1,2)], t=0.5, normal='min.max')
identif_spectra<-spectra.corr(MPdatabase, data_slopp_pivot_white[, c(1,3)], t=0.5, normal='min.max')
identif_spectra<-spectra.corr(MPdatabase, data_slopp_pivot_white[, c(1,3)], t=0.5, normal='min.max')

x <- dcast(data_slopp, "file_name + polymer +color ~ wavelength", value.var="intensity")
x$color <- as.factor(x$color)
x$polymer <- as.factor(x$polymer)
colnames(x)[4:length(colnames(x))] = paste0('w_', as.integer(colnames(x)[4:length(colnames(x))]))

glm('polymer ~ . - w_0 - w_3500 - file_name - color', data=x, family=multinomial())

N=5
c(min(x[,N]), max(x[,N]))

pc <- prcomp(x[,-c(1,2,3,4,length(x))],
             center = TRUE,
             scale. = TRUE)
#attributes(pc)
#summary(pc)

dim(t(pc$rotation))
pc_data = t(pc$rotation)
pc_data = pc$rotation

error_vector = c()

for (npc in 2:20) {
  error_vector_aux = c()
  for (i in 1:100) {
    #npc = 5
    train = sample(148, size = 148 *.7, replace = F)
    y_train <- as.character(x$polymer[train])
    data_train <- data.frame(polymer = y_train, pc_data[train,1:npc])
    
    test <- multinom(polymer ~ ., data=data_train)
    #attributes(test)
    #summary(test)
    
    error_vector_aux[i] = mean(as.character(predict(test, pc_data[-train,1:npc])) == as.character(x$polymer[-train]))
  }
  error_vector[npc] = mean(error_vector_aux)
}

plot(error_vector)




