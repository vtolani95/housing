import numpy as np
import pdb
import csv

def one_hot(num_classes, index):
  result = np.zeros(num_classes)
  if (index < 0): return result
  result[index] = 1.0
  return result

#16 long
def ms_sub_class(features, row):
  options = {
  "20" : 1,
  "30" : 2,
  "40" : 3,
  "45" : 4,
  "50" : 5,
  "60" : 6,
  "70" : 7,
  "75" : 8,
  "80" : 9,
  "85" : 10,
  "90" : 11,
  "120" : 12,
  "150" : 13,
  "160" : 14,
  "180" : 15,
  "190" : 16
  }
  features.extend(one_hot(len(options), options[row[1]]-1))

# 8 options
def ms_zoning(features, row):
  options = {
  "A" : 1,
  "C" : 2,
  "FV" : 3,
  "I" : 4,
  "RH" : 5,
  "RL" : 6,
  "RP" : 7,
  "RM" : 8,
  "C (all)": 2,
  "NA": 0
  }
  features.extend(one_hot(len(options)-2, options[row[2]]-1))


def lot_frontage(features, row):
  if row[3] == 'NA':
    features.append(0)
  else:
    features.append(float(row[3]))

def lot_area(features, row):
  features.append(np.log(float(row[4])))

def street(features, row):
  if row[5] == 'Pave':
    features.extend([0, 1])
  elif row[5] == 'Grvl':
    features.extend([1, 0])

def alley(features, row):
  if row[6] == 'NA':
    features.extend([1, 0, 0])
  elif row[6] == 'Pave':
    features.extend([0, 1, 0])
  elif row[6] == 'Grvl':
    features.extend([0, 0, 1])
  else:
    raise Exception('Invalid Alley Type')

def lot_shape(features, row):
  if row[7] == 'Reg':
    features.extend([1, 0, 0, 0])
  elif row[7] == 'IR1':
    features.extend([0, 1, 0, 0])
  elif row[7] == 'IR2':
    features.extend([0, 0, 1, 0])
  elif row[7] == 'IR3':
    features.extend([0, 0, 0, 1])
  else:
    raise Exception('Invalid Lot Shape')

def land_contour(features, row):
  if row[8] == 'Lvl':
    features.extend([1, 0, 0, 0])
  elif row[8] == 'Bnk':
    features.extend([0, 1, 0, 0])
  elif row[8] == 'HLS':
    features.extend([0, 0, 1, 0])
  elif row[8] == 'Low':
    features.extend([0, 0, 0, 1])
  else:
    raise Exception('Invalid Land Contour')

def ties(features, row):
  if row[9] == 'AllPub':
    features.extend([1, 0, 0, 0])
  elif row[9] == 'NoSewr':
    features.extend([0, 1, 0, 0])
  elif row[9] == 'NoSeWa':
    features.extend([0, 0, 1, 0])
  elif row[9] == 'ELO':
    features.extend([0, 0, 0, 1])
  elif row[9] == 'NA':
    features.extend([0, 0, 0, 0])
  else:
    raise Exception('Invalid ties')

def lot_config(features, row):
  if row[10] == 'Inside':
    features.extend([1, 0, 0, 0, 0])
  elif row[10] == 'Corner':
    features.extend([0, 1, 0, 0, 0])
  elif row[10] == 'CulDSac':
    features.extend([0, 0, 1, 0, 0])
  elif row[10] == 'FR2':
    features.extend([0, 0, 0, 1, 0])
  elif row[10] == 'FR3':
    features.extend([0, 0, 0, 0, 1])
  else:
    raise Exception('Invalid Lot Config')

def land_slope(features, row):
  if row[11] == 'Gtl':
    features.extend([1, 0, 0])
  elif row[11] == 'Mod':
    features.extend([0, 1, 0])
  elif row[11] == 'Sev':
    features.extend([0, 0, 1])
  else:
    raise Exception('Invalid Land Slope')

def neighborhood(features, row):
  neighborhoods = {
    "Blmngtn" : 1,
    "Blueste" : 2,
    "BrDale" : 3,
    "BrkSide" : 4,
    "ClearCr" : 5,
    "CollgCr" : 6,
    "Crawfor" : 7,
    "Edwards" : 8,
    "Gilbert" : 9,
    "IDOTRR" : 10,
    "MeadowV" : 11,
    "Mitchel" : 12,
    "NAmes" : 13,
    "NoRidge" : 14,
    "NPkVill" : 15,
    "NridgHt" : 16,
    "NWAmes" : 17,
    "OldTown" : 18,
    "SWISU" : 19,
    "Sawyer" : 20,
    "SawyerW" : 21,
    "Somerst" : 22,
    "StoneBr" : 23,
    "Timber" : 24,
    "Veenker" : 25
  }
  features.extend(one_hot(len(neighborhoods), neighborhoods[row[12]]-1))

def condition(features, row):
  conds = {
  "Artery": 1,
  "Feedr" : 2,
  "Norm" : 3,
  "RRNn" : 4,
  "RRAn" : 5,
  "PosN" : 6,
  "PosA" : 7,
  "RRNe" : 8,
  "RRAe" : 9
  }
  cond1 = one_hot(len(conds), conds[row[13]]-1)
  cond2 = one_hot(len(conds), conds[row[14]]-1)
  if row[13] == 'Norm' and row[14] == 'Norm':
    features.extend(cond1)
  else:
    features.extend(cond1 + cond2)

def building_type(features, row):
  types = {
  "1Fam" : 1,
  "2fmCon" : 2,
  "Duplex" : 3,
  "TwnhsE" : 4,
  "TwnhsI" : 5,
  "Twnhs" : 6
  }
  features.extend(one_hot(len(types), types[row[15]]-1))

def house_style(features, row):
  styles = {
  "1Story" : 1,
  "1.5Fin" : 2,
  "1.5Unf" : 3,
  "2Story" : 4,
  "2.5Fin" : 5,
  "2.5Unf" : 6,
  "SFoyer" : 7,
  "SLvl" : 8
  }
  features.extend(one_hot(len(styles), styles[row[16]]-1))

def overall_qual(features, row):
  features.append(float(row[17]))

def overall_cond(features, row):
  features.append(float(row[18]))

def year_built(features, row):
  features.append(float(row[19]))

def year_remod(features, row):
  features.append(float(row[20]) > 1995)

def remodeled(features, row):
  if row[19] == row[20]:
    features.append(0.0)
  else:
    features.append(1.0)

def roof_style(features, row):
  styles = {
  "Flat" : 1,
  "Gable" : 2,
  "Gambrel" : 3,
  "Hip" : 4,
  "Mansard" : 5,
  "Shed" : 6
  }
  features.extend(one_hot(len(styles), styles[row[21]]-1))

def roof_material(features, row):
  materials = {
  "ClyTile" : 1,
  "CompShg" : 2,
  "Membran" : 3,
  "Metal" : 4,
  "Roll" : 5,
  "Tar&Grv" : 6,
  "WdShake" : 7,
  "WdShngl" : 8
  }
  features.extend(one_hot(len(materials), materials[row[22]]-1))

def exterior_covering(features, row):
  coverings = {
  "AsbShng" : 1,
  "AsphShn" : 2,
  "BrkComm" : 3,
  "Brk Cmn" : 3,
  "BrkFace" : 4,
  "CBlock" : 5,
  "CmentBd" : 6,
  "CemntBd" : 6,
  "HdBoard" : 7,
  "ImStucc" : 8,
  "MetalSd" : 9,
  "Other" : 10,
  "Plywood" : 11,
  "PreCast" : 12,
  "Stone" : 13,
  "Stucco" : 14,
  "VinylSd" : 15,
  "Wd Sdng" : 16,
  "WdShing" : 17,
  "Wd Shng" : 17,
  "NA": 0
  }
  # 3 repeats and NA option = minus 4
  exterior1 = one_hot(len(coverings)-4, coverings[row[23]]-1)
  exterior2 = one_hot(len(coverings)-4, coverings[row[24]]-1)
  if row[23] == row[24]:
    features.extend(exterior1)
  else:
    features.extend(exterior1 + exterior2)

def mas_vnr_type(features, row):
  types = {
  "BrkCmn" : 1,
  "BrkFace" : 2,
  "CBlock" : 3,
  "None" : 4,
  "Stone" : 5,
  "NA" : 0
  }
  features.extend(one_hot(len(types)-1, types[row[25]]-1))

def mas_vnr_area(features, row):
  if row[26] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[26]))

def exterior_qual(features, row):
  qualities = {
  "Ex": 1,
  "Gd": 2,
  "TA": 3,
  "Fa": 4,
  "Po": 5
  }
  features.extend(one_hot(len(qualities), qualities[row[27]]-1))

def exterior_cond(features, row):
  qualities = {
  "Ex": 1,
  "Gd": 2,
  "TA": 3,
  "Fa": 4,
  "Po": 5
  }
  features.extend(one_hot(len(qualities), qualities[row[28]]-1))

def foundation(features, row):
  materials = {
  "BrkTil" : 1,
  "CBlock" : 2,
  "PConc" : 3,
  "Slab" : 4,
  "Stone" : 5,
  "Wood" : 6
  }
  features.extend(one_hot(len(materials), materials[row[29]]-1))

def basement_qual(features, row):
  qualities = {
  "Ex": 1,
  "Gd": 2,
  "TA": 3,
  "Fa": 4,
  "Po": 5,
  "NA": 0
  }
  features.extend(one_hot(len(qualities)-1, qualities[row[30]]-1))

def basement_cond(features, row):
  qualities = {
  "Ex": 1,
  "Gd": 2,
  "TA": 3,
  "Fa": 4,
  "Po": 5,
  "NA": 0
  }
  features.extend(one_hot(len(qualities)-1, qualities[row[31]]-1))

def basement_exposure(features, row):
  exposures = {
  "Gd" : 1,
  "Av" : 2,
  "Mn" : 3,
  "No": 0,
  "NA": 0
  }
  features.extend(one_hot(len(exposures)-2, exposures[row[32]]-1))

def basement_fin_1(features, row):
  options = {
  "GLQ" : 1,
  "ALQ" : 2,
  "BLQ" : 3,
  "Rec" : 4,
  "LwQ" : 5,
  "Unf" : 6,
  "NA" : 0
  }
  features.extend(one_hot(len(options)-1, options[row[33]]-1))

def basement_1_sf(features, row):
  if row[34] == 'NA':
    features.append(0.0)
  else:
    features.append(float(row[34]))

def basement_fin_2(features, row):
  options = {
  "GLQ" : 1,
  "ALQ" : 2,
  "BLQ" : 3,
  "Rec" : 4,
  "LwQ" : 5,
  "Unf" : 6,
  "NA" : 0
  }
  features.extend(one_hot(len(options)-1, options[row[35]]-1))

def basement_2_sf(features, row):
  if row[36] == 'NA':
    features.append(0.0)
  else:
    features.append(float(row[36]))

## unsure if good or bad to log here
def basement_unfn_sf(features, row):
  if row[37] == 'NA':
    features.append(0.0)
  elif row[37] == "0":
    features.append(0.0)
  else:
    features.append(float(row[37]))

def basement_total_sq_ft(features, row):
  if row[38] == "NA":
    features.append(0.0)
  elif row[38] == "0":
    features.append(0.0)
  else:
    features.append(np.log(float(row[38])))

def heating(features, row):
  options = {
  "Floor" : 1,
  "GasA" : 2,
  "GasW" : 3,
  "Grav" : 4,
  "OthW" : 5,
  "Wall" : 6
  }
  features.extend(one_hot(len(options), options[row[39]]-1))

def heating_qc(features, row):
  qualities = {
  "Ex": 1,
  "Gd": 2,
  "TA": 3,
  "Fa": 4,
  "Po": 5,
  "NA": 0
  }
  features.extend(one_hot(len(qualities)-1, qualities[row[40]]-1))

def central_air(features, row):
  if row[41] == 'Y':
    features.append(1.0)
  elif row[41] == 'N':
    features.append(0.0)
  else:
    raise Exception('Invalid Central Air Type')

def electrical_system(features, row):
  options = {
  "SBrkr" : 1,
  "FuseA" : 2,
  "FuseF"  : 3,
  "FuseP" : 4,
  "Mix" : 5,
  "NA" : 0
  }
  features.extend(one_hot(len(options)-1, options[row[42]]-1))

def first_floor_sq_ft(features, row):
  features.append(float(row[43]))

def second_floor_sq_ft(features, row):
  features.append(float(row[43]))

def low_quality_fin_sf(features, row):
  features.append(float(row[45]))

#not sure if log is good or not
def gr_live_area(features, row):
  features.append(float(row[46]))

def bsmnt_full_bath(features, row):
  if row[47] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[47]))

def bsmnt_half_bath(features, row):
  if row[48] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[48]))

def full_bath(features, row):
  if row[49] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[49]))

def half_bath(features, row):
  if row[50] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[50]))

def bedroom_abv_grd(features, row):
  if row[51] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[51]))

def kitchen_abv_grd(features, row):
  if row[52] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[52]))

def kitchen_quality(features, row):
  qualities = {
  "Ex": 1,
  "Gd": 2,
  "TA": 3,
  "Fa": 4,
  "Po": 5,
  "NA": 0
  }
  features.extend(one_hot(len(qualities)-1, qualities[row[53]]-1))

def total_rooms_abv_grd(features, row):
  if row[54] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[54]))

def functionality(features, row):
  options = {
  "Typ" : 1,
  "Min1" : 2,
  "Min2" : 3,
  "Mod" : 4,
  "Maj1" : 5,
  "Maj2" : 6,
  "Sev" : 7,
  "Sal" : 8,
  "NA" : 0
  }
  features.extend(one_hot(len(options)-1, options[row[55]]-1))

def fireplaces(features, row):
  if row[56] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[56]))

def fireplace_quality(features, row):
  qualities = {
  "Ex": 1,
  "Gd": 2,
  "TA": 3,
  "Fa": 4,
  "Po": 5,
  "NA": 0
  }
  features.extend(one_hot(len(qualities)-1, qualities[row[57]]-1))

def garage_type(features, row):
  options = {
  "2Types" : 1,
  "Attchd" : 2,
  "Basment" : 3,
  "BuiltIn" : 4,
  "CarPort" : 5,
  "Detchd" : 6,
  "NA": 0
  }
  features.extend(one_hot(len(options)-1, options[row[58]]-1))

def garage_yr_built(features, row):
  if row[59] == 'NA':
    features.append(0.0)
  else:
    features.append(float(row[59]))

def garage_finish(features, row):
  options = {
  "Fin" : 1,
  "RFn" : 2,
  "Unf" : 3,
  "NA" : 0
  }
  features.extend(one_hot(len(options)-1, options[row[60]]-1))

def garage_cars(features, row):
  if row[61] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[61]))

def garage_area(features, row):
  if row[62] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[62]))

def garage_quality(features, row):
  qualities = {
  "Ex": 1,
  "Gd": 2,
  "TA": 3,
  "Fa": 4,
  "Po": 5,
  "NA": 0
  }
  features.extend(one_hot(len(qualities)-1, qualities[row[63]]-1))

def garage_cond(features, row):
  qualities = {
  "Ex": 1,
  "Gd": 2,
  "TA": 3,
  "Fa": 4,
  "Po": 5,
  "NA": 0
  }
  features.extend(one_hot(len(qualities)-1, qualities[row[64]]-1))

def paved_driveway(features, row):
  options = {
  "Y" : 1,
  "P" : 2,
  "N" : 0
  }
  features.extend(one_hot(len(options), options[row[65]]-1))

def wood_deck_area(features, row):
  features.append(float(row[66]))

def open_porch_sf(features, row):
  features.append(float(row[67]))

def encolosed_porch(features, row):
  features.append(float(row[68]))

def three_season_porch(features, row):
  features.append(float(row[69]))

def screen_porch(features, row):
  features.append(float(row[70]))

def pool_area(features, row):
  features.append(float(row[71]))

def pool_quality(features, row):
  qualities = {
  "Ex": 1,
  "Gd": 2,
  "TA": 3,
  "Fa": 4,
  "Po": 5,
  "NA": 0
  }
  features.extend(one_hot(len(qualities)-1, qualities[row[72]]-1))

def fence_quality(features, row):
  qualities = {
  "GdPrv" : 1,
  "MnPrv" : 2,
  "GdWo" : 3,
  "MnWw" : 4,
  "NA" : 5 #no fence could be a drawback
  }
  features.extend(one_hot(len(qualities), qualities[row[73]]-1))

def misc_feature(features, row):
  options = {
  "Elev" : 1,
  "Gar2" : 2,
  "Othr" : 3,
  "Shed" : 4,
  "TenC" : 5,
  "NA" : 0
  }
  features.extend(one_hot(len(options)-1, options[row[74]]-1))

def misc_val(features, row):
  if row[75] == "NA":
    features.append(0.0)
  else:
    features.append(float(row[75]))

def month_sold(features, row):
  features.append(float(row[76]))

def year_sold(features, row):
  # features.append(float(row[77]))
  year = float(row[77])
  if year >= 2006 and year < 2009:
    features.append(1.0)
  else:
    features.append(0.0)

def sale_type(features, row):
  options = {
  "WD" : 1,
  "CWD" : 2,
  "VWD" : 3,
  "New" : 4,
  "COD" : 5,
  "Con" : 6,
  "ConLw" : 7,
  "ConLI": 8,
  "ConLD": 9,
  "Oth" : 0, #multiple different types so ignore this
  "NA" : 0
  }
  features.extend(one_hot(len(options)-2, options[row[78]]-1))

def sale_condition(features, row):
  options = {
  "Normal" : 1,
  "Abnorml" : 2,
  "AdjLand" : 3,
  "Alloca" : 4,
  "Family" : 5,
  "Partial" : 6
  }
  features.extend(one_hot(len(options), options[row[79]]-1))


def load_data(filename, train=True):
  with open(filename, newline='') as csvfile:
    x = []
    y = []
    reader = csv.reader(csvfile)
    flag = 0
    skip = set(['524', '1299'])
    for row in reader:
      if flag == 0:
        flag = 1
        continue
      if row[0] in skip: #outliers
        continue
      if train:
        y.append(np.log(float(row[-1])))
      else:
        y.append(int(row[0]))
      features = []

      lot_frontage(features, row)
      lot_area(features, row)
      mas_vnr_area(features, row) #

      ms_sub_class(features, row)
      ms_zoning(features, row)
      street(features, row)
      # alley(features, row)
      lot_shape(features, row)
      land_contour(features, row)
      ties(features, row)
      lot_config(features, row)
      land_slope(features, row)
      neighborhood(features, row)
      condition(features, row)
      building_type(features, row)
      house_style(features, row)
      overall_qual(features, row)
      overall_cond(features, row)
      year_built(features, row)
      year_remod(features, row)
      # remodeled(features, row)
      roof_style(features, row)
      roof_material(features, row)
      exterior_covering(features, row)
      mas_vnr_type(features, row)

      exterior_qual(features, row)
      exterior_cond(features, row)
      foundation(features, row)
      basement_qual(features, row)
      basement_cond(features, row)
      basement_exposure(features, row)
      basement_fin_1(features, row)
      basement_1_sf(features, row) #
      basement_fin_2(features, row)
      basement_2_sf(features, row) #
      basement_unfn_sf(features, row)
      basement_total_sq_ft(features, row)
      heating(features, row)
      heating_qc(features, row) #
      central_air(features, row)
      electrical_system(features, row)
      first_floor_sq_ft(features, row)
      second_floor_sq_ft(features, row) #
      low_quality_fin_sf(features, row) #
      gr_live_area(features, row)
      bsmnt_full_bath(features, row)
      bsmnt_half_bath(features, row) #
      full_bath(features, row)
      half_bath(features, row)
      bedroom_abv_grd(features, row)
      kitchen_abv_grd(features, row)
      kitchen_quality(features, row)
      total_rooms_abv_grd(features, row)
      functionality(features, row)
      fireplaces(features, row)
      # fireplace_quality(features, row)
      garage_type(features, row)
      garage_yr_built(features, row)
      garage_finish(features, row)
      garage_cars(features, row)
      garage_area(features, row)
      garage_quality(features, row)
      garage_cond(features, row)
      paved_driveway(features, row)
      wood_deck_area(features, row)
      open_porch_sf(features, row) #
      encolosed_porch(features, row)
      three_season_porch(features, row)
      screen_porch(features, row)
      pool_area(features, row)
      pool_quality(features, row)
      # pool(features, row)
      fence_quality(features, row)
      misc_feature(features, row)
      misc_val(features, row)
      month_sold(features, row) #
      year_sold(features, row) #
      sale_type(features, row)
      sale_condition(features, row)

      x.append(features)
  return np.array(x), np.array(y)

def write_csv(filename, id, pred):
  c = csv.writer(open(filename, "wt"))
  c.writerow(['Id', 'SalePrice'])
  for i in range(len(pred)):
    c.writerow((id[i], pred[i]))






