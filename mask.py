#!/usr/bin/python3

import os
from transformers import BertForMaskedLM, BertTokenizerFast, AutoModelForMaskedLM, AutoTokenizer, LongformerForMaskedLM
import torch
import csv

try:
    from transformers.modeling_longformer import LongformerSelfAttention
except ImportError:
    from transformers import LongformerSelfAttention

class BertLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask,
                output_attentions=output_attentions)


class BertLong(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.bert.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with
            # `LongformerSelfAttention`
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)

#model_path = '/Users/gijsdanoe/Documents/Informatiekunde/Masterscriptie/longformer/bert-base-dutch-4096'
#model_path = '/Users/gijsdanoe/Documents/Informatiekunde/Masterscriptie/robbert-long/robbert-v2-dutch-base-4096'
#model_path = 'GroNLP/bert-base-dutch-cased'
#model_path = 'pdelobelle/robbert-v2-dutch-base'
#model_path = 'markussagen/xlm-roberta-longformer-base-4096'
#model_path = 'flax-community/pino-bigbird-roberta-base'

if model_path == '/Users/gijsdanoe/Documents/Informatiekunde/Masterscriptie/robbert-long/robbert-v2-dutch-base-4096':
	tokenizer_path = 'pdelobelle/robbert-v2-dutch-base'
else:
	tokenizer_path = model_path


tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)



short1 = """
BEREIDING ROTI
Kook de [MASK] in 8 minuten hard, zodat je hier straks niet op hoeft te wachten. Laat ze schrikken zodra ze klaar zijn. Zet apart. 
Was de [MASK] en dep het droog. Zet een wok of grote hapjespan op het vuur en verwarm daarin de [MASK]. Snipper de [MASK], pers 2 teentjes [MASK] uit en voeg alles toe aan de pan. Fruit een paar minuutjes op laag vuur.
Zodra de [MASK] glazig is, voeg je 2 theelepels [MASK] toe. Roer goed door de pan en strooi 1 eetlepel [MASK] over het mengsel. Laat een minuut sudderen.
Voeg de [MASK] toe, samen met wat [MASK] en [MASK]. Bak al roerende de [MASK] een beetje aan. Verkruimel een half blokje [MASK] en strooi dit over de [MASK]. Voeg een half kopje – ongeveer 100 ml – [MASK] toe. Doe een deksel op de pan en laat op laag vuur 30 minuten stoven. Bekijk af en toe of er niet zoveel [MASK] is verdampt dat de [MASK] aanbrandt. Voeg eventueel nog wat [MASK] toe. 
Terwijl de [MASK] lekker staat te pruttelen, beginnen we aan de [MASK]. Schil deze en snijd ze in blokken van ongeveer 4 bij 4 cm.
Neem een hapjes- of koekenpan en verhit hierin 2 eetlepels [MASK]. Doe de [MASK] en [MASK] in de pan en bak tot ze glazig zijn. 
Doe de [MASK] erbij, samen met de [MASK]. Meng alles goed op een middelhoog vuur. Voeg de blokjes [MASK], een half blokje verkruimelde [MASK] en 100 milliliter [MASK] toe en laat 15 minuten zachtjes stoven. Blijf ook nu goed opletten dat er niet teveel [MASK] verdampt.
Snijd de [MASK] in steeltjes van ongeveer 10 cm. Bak de [MASK] met een halve gesnipperde [MASK] in wat [MASK] en maak het op smaak met een teentje geperste [MASK], de [MASK], [MASK], [MASK], wat [MASK] en [MASK]. Proef of de [MASK] gaar is.
Pel de hardgekookte [MASK]. Frituur de gepelde [MASK] in een ruime hoeveelheid [MASK] (de [MASK] moeten net onder staan). Als ze mooi bruin zijn voeg je de [MASK], [MASK] en [MASK] toe. Roer alles een paar keer goed door en laat de [MASK] uitlekken in een zeef of vergiet met keukenpapier. 
"""

short2 = """
BEREIDING ROTI
Kook de <mask> in 8 minuten hard, zodat je hier straks niet op hoeft te wachten. Laat ze schrikken zodra ze klaar zijn. Zet apart. 
Was de <mask> en dep het droog. Zet een wok of grote hapjespan op het vuur en verwarm daarin de <mask>. Snipper de <mask>, pers 2 teentjes <mask> uit en voeg alles toe aan de pan. Fruit een paar minuutjes op laag vuur.
Zodra de <mask> glazig is, voeg je 2 theelepels <mask> toe. Roer goed door de pan en strooi 1 eetlepel <mask> over het mengsel. Laat een minuut sudderen.
Voeg de <mask> toe, samen met wat <mask> en <mask>. Bak al roerende de <mask> een beetje aan. Verkruimel een half blokje <mask> en strooi dit over de <mask>. Voeg een half kopje – ongeveer 100 ml – <mask> toe. Doe een deksel op de pan en laat op laag vuur 30 minuten stoven. Bekijk af en toe of er niet zoveel <mask> is verdampt dat de <mask> aanbrandt. Voeg eventueel nog wat <mask> toe. 
Terwijl de <mask> lekker staat te pruttelen, beginnen we aan de <mask>. Schil deze en snijd ze in blokken van ongeveer 4 bij 4 cm.
Neem een hapjes- of koekenpan en verhit hierin 2 eetlepels <mask>. Doe de <mask> en <mask> in de pan en bak tot ze glazig zijn. 
Doe de <mask> erbij, samen met de <mask>. Meng alles goed op een middelhoog vuur. Voeg de blokjes <mask>, een half blokje verkruimelde <mask> en 100 milliliter <mask> toe en laat 15 minuten zachtjes stoven. Blijf ook nu goed opletten dat er niet teveel <mask> verdampt.
Snijd de <mask> in steeltjes van ongeveer 10 cm. Bak de <mask> met een halve gesnipperde <mask> in wat <mask> en maak het op smaak met een teentje geperste <mask>, de <mask>, <mask>, <mask>, wat <mask> en <mask>. Proef of de <mask> gaar is.
Pel de hardgekookte <mask>. Frituur de gepelde <mask> in een ruime hoeveelheid <mask> (de <mask> moeten net onder staan). Als ze mooi bruin zijn voeg je de <mask>, <mask> en <mask> toe. Roer alles een paar keer goed door en laat de <mask> uitlekken in een zeef of vergiet met keukenpapier.
"""


long1 = """
INGREDIËNTEN: 2 el olie, kip, 8 kippendijen met bot (het botje houdt de kip sappig), 1 ui, 2 teentjes knoflook, 2 theelepels tomatenpuree, 1 eetlepel masala, peper en zout, 1⁄2 blokje bouillon, 100 ml water, 4 eieren, 1 ui (gesnipperd), 1 teentje knoflook (geperst), 200 gram kouseband, 1⁄2 theelepel gemberpoeder, 1⁄2 theelepel laos, 1⁄2 theelepel kerriepoeder, 4 grote aardappelen, 1⁄2 theelepel komijnpoeder, 500 gram zelfrijzend bakmeel, 350 ml lauwwarm water
BEREIDING ROTI
Kook de [MASK] in 8 minuten hard, zodat je hier straks niet op hoeft te wachten. Laat ze schrikken zodra ze klaar zijn. Zet apart. 
Was de [MASK] en dep het droog. Zet een wok of grote hapjespan op het vuur en verwarm daarin de [MASK]. Snipper de [MASK], pers 2 teentjes [MASK] uit en voeg alles toe aan de pan. Fruit een paar minuutjes op laag vuur.
Zodra de [MASK] glazig is, voeg je 2 theelepels [MASK] toe. Roer goed door de pan en strooi 1 eetlepel [MASK] over het mengsel. Laat een minuut sudderen.
Voeg de [MASK] toe, samen met wat [MASK] en [MASK]. Bak al roerende de [MASK] een beetje aan. Verkruimel een half blokje [MASK] en strooi dit over de [MASK]. Voeg een half kopje – ongeveer 100 ml – [MASK] toe. Doe een deksel op de pan en laat op laag vuur 30 minuten stoven. Bekijk af en toe of er niet zoveel [MASK] is verdampt dat de [MASK] aanbrandt. Voeg eventueel nog wat [MASK] toe. 
Terwijl de [MASK] lekker staat te pruttelen, beginnen we aan de [MASK]. Schil deze en snijd ze in blokken van ongeveer 4 bij 4 cm.
Neem een hapjes- of koekenpan en verhit hierin 2 eetlepels [MASK]. Doe de [MASK] en [MASK] in de pan en bak tot ze glazig zijn. 
Doe de [MASK] erbij, samen met de [MASK]. Meng alles goed op een middelhoog vuur. Voeg de blokjes [MASK], een half blokje verkruimelde [MASK] en 100 milliliter [MASK] toe en laat 15 minuten zachtjes stoven. Blijf ook nu goed opletten dat er niet teveel [MASK] verdampt.
Snijd de [MASK] in steeltjes van ongeveer 10 cm. Bak de [MASK] met een halve gesnipperde [MASK] in wat [MASK] en maak het op smaak met een teentje geperste [MASK], de [MASK], [MASK], [MASK], wat [MASK] en [MASK]. Proef of de [MASK] gaar is.
Pel de hardgekookte [MASK]. Frituur de gepelde [MASK] in een ruime hoeveelheid [MASK] (de [MASK] moeten net onder staan). Als ze mooi bruin zijn voeg je de [MASK], [MASK] en [MASK] toe. Roer alles een paar keer goed door en laat de [MASK] uitlekken in een zeef of vergiet met keukenpapier. 
"""

long2 = """
Echte Surinaamse Roti Maken [Recept]
INGREDIËNTEN: 2 el olie, kip, 8 kippendijen met bot (het botje houdt de kip sappig), 1 ui, 2 teentjes knoflook, 2 theelepels tomatenpuree, 1 eetlepel masala, peper en zout, 1⁄2 blokje bouillon, 100 ml water, 4 eieren, 1 ui (gesnipperd), 1 teentje knoflook (geperst), 200 gram kouseband, 1⁄2 theelepel gemberpoeder, 1⁄2 theelepel laos, 1⁄2 theelepel kerriepoeder, 4 grote aardappelen, 1⁄2 theelepel komijnpoeder, 500 gram zelfrijzend bakmeel, 350 ml lauwwarm water
BEREIDING ROTI
Kook de <mask> in 8 minuten hard, zodat je hier straks niet op hoeft te wachten. Laat ze schrikken zodra ze klaar zijn. Zet apart. 
Was de <mask> en dep het droog. Zet een wok of grote hapjespan op het vuur en verwarm daarin de <mask>. Snipper de <mask>, pers 2 teentjes <mask> uit en voeg alles toe aan de pan. Fruit een paar minuutjes op laag vuur.
Zodra de <mask> glazig is, voeg je 2 theelepels <mask> toe. Roer goed door de pan en strooi 1 eetlepel <mask> over het mengsel. Laat een minuut sudderen.
Voeg de <mask> toe, samen met wat <mask> en <mask>. Bak al roerende de <mask> een beetje aan. Verkruimel een half blokje <mask> en strooi dit over de <mask>. Voeg een half kopje – ongeveer 100 ml – <mask> toe. Doe een deksel op de pan en laat op laag vuur 30 minuten stoven. Bekijk af en toe of er niet zoveel <mask> is verdampt dat de <mask> aanbrandt. Voeg eventueel nog wat <mask> toe. 
Terwijl de <mask> lekker staat te pruttelen, beginnen we aan de <mask>. Schil deze en snijd ze in blokken van ongeveer 4 bij 4 cm.
Neem een hapjes- of koekenpan en verhit hierin 2 eetlepels <mask>. Doe de <mask> en <mask> in de pan en bak tot ze glazig zijn. 
Doe de <mask> erbij, samen met de <mask>. Meng alles goed op een middelhoog vuur. Voeg de blokjes <mask>, een half blokje verkruimelde <mask> en 100 milliliter <mask> toe en laat 15 minuten zachtjes stoven. Blijf ook nu goed opletten dat er niet teveel <mask> verdampt.
Snijd de <mask> in steeltjes van ongeveer 10 cm. Bak de <mask> met een halve gesnipperde <mask> in wat <mask> en maak het op smaak met een teentje geperste <mask>, de <mask>, <mask>, <mask>, wat <mask> en <mask>. Proef of de <mask> gaar is.
Pel de hardgekookte <mask>. Frituur de gepelde <mask> in een ruime hoeveelheid <mask> (de <mask> moeten net onder staan). Als ze mooi bruin zijn voeg je de <mask>, <mask> en <mask> toe. Roer alles een paar keer goed door en laat de <mask> uitlekken in een zeef of vergiet met keukenpapier. 
"""
gold = ['eieren','kip','olie,','ui','knoflook', 'ui','tomatenpuree','masala','kip','zout','peper','kip','bouillon','kip','water','water','kip','water','kip','aardappelen','olie','ui','knoflook','tomatenpuree','masala','aardappelen','bouillon','water','water','kouseband','kouseband','ui','olie','knoflook','gemberpoeder','laos','kerriepoeder','peper','zout','kouseband','eieren','eieren','olie','eieren','kerriepoeder','gember','laos','eieren']



inputs = tokenizer(long2, return_tensors='pt', max_length=4096)
print(len(inputs[0]))
with torch.no_grad():
    logits = model(**inputs)[0]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
predicted = tokenizer.decode(predicted_token_id).split()

score = 0
for x,y in zip(predicted, gold):
	if x == y:
		score += 1

print(score, '/', len(gold))
print(predicted)
