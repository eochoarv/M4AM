interpolate_spectra <- function(data, min=0, max=3500, n=1400, normalice=NA){
  
  if (is.na(normalice)) {
    data_norm = data
  } else {
    if (normalice=='SNV') {
      data_norm = RamanMP::norm.SNV(data)
    } else {
      data_norm = RamanMP::norm.min.max(data)
    }
  }
  
  data_norm_long <- melt(data_norm, id="freq")
  colnames(data_norm_long) <- c('wavelength', 'file_name', 'intensity')
  data_file_list <- as.character(unique(data_norm_long$file_name))
  
  min_max_data <- data_norm_long %>% 
    group_by(file_name) %>% 
    summarise(wavelength_min = min(wavelength), 
              wavelength_max = max(wavelength)) %>% 
    ungroup()
  
  polymer_short_names = c('PEVA',
                          'Nylon',
                          'PVC',
                          'PE',
                          'PET',
                          'PLA',
                          'PS',
                          'Acrylic',
                          'Cotton',
                          'Polyester',
                          'ABS',
                          'PU',
                          'PEA',
                          'R',
                          'PC',
                          'PP',
                          'PA',
                          'PMMA',
                          'PVA',
                          'PBT')
  
  polymer_names = c('Polyethylene Vinyl Acetate',
                    'Nylon',
                    'Polyvinyl Chloride', 
                    'Polyethylene', 
                    'Polyethylene Terephthalate', 
                    'Polylactic acid', 
                    'Polystyrene', 
                    'Acrylic', 
                    'Cotton', 
                    'Polyester', 
                    'Acrylonitrile Butadiene Styrene', 
                    'Polyurethane', 
                    'Polyethylene Adipate', 
                    'Styrene-butadiene Rubber', 
                    'Polycarbonate', 
                    'Polypropylene',
                    'Polyamide',
                    'Polymethyl Methacrylate',
                    'Polyvinyl Acetate',
                    'Polybutylene Terephthalate')
  
  dict_pol_names = data.frame(pol_short_names=polymer_short_names, polymer=polymer_names)
  
  polymer_list = c()
  for (fn in data_file_list) {
    polymer = substr(fn, 1, unlist(gregexpr("_", fn))-1)
    if (polymer=="") {
      polymer = substr(fn, 1, 2)
    }
    else{
      if (substr(polymer,nchar(polymer),nchar(polymer)) %in% as.character(0:9)) {
        polymer = substr(fn, 1, 2)
      }
    }
    polymer_list = c(polymer_list, polymer)
  } 
  
  polymer_df = data.frame(file_name=data_file_list, 
                          pol_short_names=polymer_list)
  
  polymer_df = left_join(polymer_df, dict_pol_names, by="pol_short_names") %>%
    select(-c(pol_short_names))
  
  #polymer_list = unique(polymer_list)
  
  all_wl = seq(0, 3500, length.out = 1400) 
  
  data_inter_norm = data.frame()
  
  for(fn in data_file_list){
    data_fn = data_norm_long %>% filter(file_name == fn)
    df_lenght = dim(data_fn)[1]
    min_wl = (min_max_data %>% filter(file_name == fn))$wavelength_min
    max_wl = (min_max_data %>% filter(file_name == fn))$wavelength_max
    
    x = data_fn[,'wavelength']
    y = ifelse(is.na(data_fn[,'intensity']), 0, data_fn[,'intensity'])
    f <- approxfun(x, y)
    
    x_new = all_wl[all_wl >= min_wl & all_wl <= max_wl]
    x_out = all_wl[all_wl < min_wl | all_wl > max_wl]
    
    data_inter_norm = rbind(data_inter_norm, data.frame(wavelength=c(x_new, x_out),
                                                                    intensity=c(f(x_new), rep(0, length(x_out))),
                                                                    file_name=rep(fn, 1400)))
  }
  
  data_inter_norm = left_join(data_inter_norm, polymer_df, by="file_name")
  return(data_inter_norm)
}
