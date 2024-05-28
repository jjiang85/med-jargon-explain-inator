#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

inp_term="term_to_cui.json"  #input terms/cui json file
defs="cui_to_def.json"
s_data="jargon.json" 		#secondary data set to combine
termF="term_to_cui_final.json"		#final output json file
defF="cui_to_def_final.json"

def load_files(inp_file):
	with open(inp_file, 'r') as f:		#opens and loads terms/cui
		loaded_file=json.load(f)
	f.close()
	return loaded_file
	
def main():
	terms_cui=load_files(inp_term)		#input terms/cui json file
	sec_data=load_files(s_data)			#MedJ and Medline Plus set
	def_data=load_files(defs)			#cui to defintions
	
	new_terms_cui={}
	terms_removed=0
	term_count=0
	keys_to_remove=["+", ">", "\\", "kit kit", "(", "&", "_", ":", "/", "mouse"]
	
	for k in terms_cui.keys():
		if any(sub in k for sub in keys_to_remove):
			 terms_removed +=1
		else:
			new_terms_cui[k]=terms_cui[k]
			term_count +=1
	
	print("Terms Removed: " + str(terms_removed))
	print("Total Term Count: " + str(term_count))
			

	cui=0
	origin="MedJ"
	for key, value in sec_data.items():
		if key in new_terms_cui:
			continue
		else:
			cui+=1 
			ncui="M" + str(cui)
			new_terms_cui[key]=[ncui]
			if sec_data[key]:
				def_data[ncui]={}
				def_data[ncui][origin]=value
			
	with open(termF, 'w') as f:							
		f.write(json.dumps(new_terms_cui, indent=1))
	f.close()
	
	with open(defF, 'w') as d:
		d.write(json.dumps(def_data, indent=1))
	d.close()
			

if __name__=="__main__":
	main()
