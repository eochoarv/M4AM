library(RamanMP)

data("MPdatabase")
data_slopp <- read.csv('~/Documents/Research_Abuj/spectral_libraries/df_slopp_inter.csv')
data_slopp_e <- read.csv('~/Documents/Research_Abuj/spectral_libraries/df_slopp_e_inter.csv')
data_slopp_pivot <- read.csv('~/Documents/Research_Abuj/spectral_libraries/df_slopp_inter_pivot.csv')
data_slopp_pivot <- data_slopp_pivot %>% select(c(freq=wavelength, colnames(data_slopp_pivot)[2:150]))
data_slopp$file_name_2 = gsub(" ", ".", data_slopp$file_name)
data_slopp[data_slopp$file_name == "Polyurethane 17. Clear Thermo-Polyurethane Square.txt", 7] = "Polyurethane.17..Clear.Thermo.Polyurethane.Square.txt"

data_slopp_pivot_norm <- norm.SNV(data_slopp_pivot)

data_slopp_norm <- melt(data_slopp_pivot_norm, id="freq")
data_slopp_norm <- data_slopp_norm %>% select(c(wavelength=freq, intensity_norm=value, file_name_2=variable))
data_slopp_norm = left_join(data_slopp, data_slopp_norm, by=c("file_name_2", "wavelength"))

data_slopp_norm <- data_slopp_norm %>% 
  select(-c(file_name_2, intensity)) %>%
  select(wavelength, intensity=intensity_norm, polymer, file_name, color)

data_slopp_pivot_w <- dcast(data_slopp, "file_name + polymer +color ~ wavelength", value.var="intensity")
data_slopp_pivot_w$color <- as.factor(data_slopp_pivot_w$color)
data_slopp_pivot_w$polymer <- as.factor(data_slopp_pivot_w$polymer)
colnames(data_slopp_pivot_w)[4:length(colnames(data_slopp_pivot_w))] = paste0('w_', as.integer(colnames(data_slopp_pivot_w)[4:length(colnames(data_slopp_pivot_w))]))

data_slopp_norm_pivot_w <- dcast(data_slopp_norm, "file_name + polymer +color ~ wavelength", value.var="intensity")
data_slopp_norm_pivot_w$color <- as.factor(data_slopp_norm_pivot_w$color)
data_slopp_norm_pivot_w$polymer <- as.factor(data_slopp_norm_pivot_w$polymer)
colnames(data_slopp_norm_pivot_w)[4:length(colnames(data_slopp_norm_pivot_w))] = paste0('w_', as.integer(colnames(data_slopp_norm_pivot_w)[4:length(colnames(data_slopp_norm_pivot_w))]))

slopp_polymer_list <- unique(as.character(data_slopp_norm_pivot_w$polymer))
slopp_polymer_list = slopp_polymer_list[slopp_polymer_list != "Polymethyl Methacrylate"]

slopp_polymer_dict = data.frame(class=1:(length(slopp_polymer_list)), 
                                polymer=slopp_polymer_list)


MPdatabase_inter_norm = interpolate_spectra(MPdatabase, min=0, max=3500, n=1400, normalice="SNV")


MPdatabase_inter_norm_pivot_w <- dcast(MPdatabase_inter_norm, "file_name + polymer ~ wavelength", value.var="intensity")
MPdatabase_inter_norm_pivot_w$polymer <- as.factor(MPdatabase_inter_norm_pivot_w$polymer)
colnames(MPdatabase_inter_norm_pivot_w)[3:length(colnames(MPdatabase_inter_norm_pivot_w))] = paste0('w_', as.integer(colnames(MPdatabase_inter_norm_pivot_w)[3:length(colnames(MPdatabase_inter_norm_pivot_w))]))

MPdatabase_inter_norm_pivot_w_polymers = MPdatabase_inter_norm_pivot_w %>% filter(!is.na(polymer))



x <- data_slopp_norm_pivot_w[,-c(1,2,3,4,length(data_slopp_norm_pivot_w))]
x <- x[as.character(data_slopp_norm_pivot_w$polymer) %in% slopp_polymer_dict$polymer,]

y <- merge(data.frame(polymer=data_slopp_norm_pivot_w$polymer), 
           slopp_polymer_dict, by="polymer")$class

train = sample(length(y), size = length(y) *0.8, replace = F)
x_train = x[train,]
y_train = y[train]

cvfit <- cv.glmnet(x=as.matrix(x_train), y=y_train, family = "multinomial", type.multinomial = "grouped")
plot(cvfit)

#fit <- glmnet(x_train, y_train, family = "multinomial", type.multinomial = "grouped")
#plot(fit, xvar = "lambda", label = TRUE, type.coef = "2norm")

x_mat_tets <- x[-train,]
y_test = y[-train]

#y_pred = predict(fit, newx = as.matrix(x[-train,]), s = 0.000001, type = "class")[,1]
#mean(y_test == y_pred)
#y_pred = predict(fit, newx = as.matrix(x[-train,]), s = 0.00001, type = "class")[,1]
#mean(y_test == y_pred)
y_pred = predict(cvfit, newx = as.matrix(x[-train,]), s = "lambda.min", type = "class")
mean(y_test == y_pred)

table(as.numeric(y_pred), y_test)


x_mat_tets <- as.matrix(MPdatabase_inter_norm_pivot_w_polymers[,-c(1,2,3,1402)])
y_test <- as.character(MPdatabase_inter_norm_pivot_w_polymers[, 2])

y_test_2 = left_join(data.frame(polymer=as.character(y_test)), 
                     slopp_polymer_dict, by="polymer")$class

y_pred = predict(cvfit, newx = x_mat_tets, s = "lambda.min", type = "class")
mean(y_test_2 == y_pred, na.rm=T)

y_pred_prob <- predict(cvfit, newx = x_mat_tets, s = "lambda.min", type = "response")
y_pred_prob <- apply(y_pred_prob, MARGIN=1, FUN=max)

table(y_test_2, as.numeric(y_pred))

identif_spectra_polymer <- c()
identif_spectra_corr <- c()
i = 1
for (fn in MPdatabase_inter_norm_pivot_w_polymers$file_name) {
  identif_spectra <- spectra.corr(data_slopp_pivot, MPdatabase[, c('freq',fn)], t=0.5, normal='SNV')
  identif_spectra_polymer[i] = rownames(identif_spectra)[1]
  identif_spectra_corr[i] = identif_spectra[1]
  i = i + 1
}


corr_table <- left_join(data.frame(file_name_2=identif_spectra_polymer,
                     corr=identif_spectra_corr,
                     y_test),
          as.data.frame(data_slopp %>% 
                          select(file_name_2, polymer) %>% 
                          group_by(file_name_2, polymer) %>% 
                          summarise(n=n())), by='file_name_2') %>%
  select(polymer, y_test, corr) %>%
  mutate(comparision=(polymer == y_test)*1)

mean(corr_table$comparision)

prob_table <- left_join(data.frame(class=as.integer(y_pred),
                     prob=y_pred_prob,
                     y_test) ,
          slopp_polymer_dict, by='class') %>%
  select(polymer, y_test, prob) %>%
  mutate(comparision=(polymer == y_test)*1)

mean(prob_table$comparision)
