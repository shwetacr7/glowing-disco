#!/usr/bin/env python
# coding: utf-8

# In[1]:


#hide
# !pip install -Uqq fastbook
import fastbook
fastbook.setup_book()


# In[2]:


#hide
from fastbook import *
from fastai.vision.widgets import *


# In[3]:


key = os.environ.get('AZURE_SEARCH_KEY', '37d368fa7e8d4d9f99221f20fd8b2ac1')


# In[4]:


search_images_bing


# In[5]:


def search_images_bing(key, term, max_images: int = 1000, **kwargs):    
    params = {'q':term, 'count':max_images}
    headers = {"Ocp-Apim-Subscription-Key":key}
    search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()    
    return L(search_results['value'])


# In[ ]:


results = search_images_bing(key, 'senecio', min_sz=500)
ims = results.attrgot('contentUrl')


# In[ ]:


len(ims)


# In[ ]:


ims


# In[ ]:


#ims = ['https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Senecio_lautus_kz02.jpg/1200px-Senecio_lautus_kz02.jpg]


# In[ ]:


#dest = 'images/youtube5.jpg'
#download_url(ims2[0], dest)


# In[ ]:


#im = Image.open(dest)
#im.to_thumb(128,128)


# In[6]:


import os, re, os.path
mypath = "plants"
for root, dirs, files in os.walk(mypath):
    for file in files:
        os.remove(os.path.join(root, file))


# In[7]:


plant_types = 'pearl', 'banana'
path = Path('plants')


# In[8]:


if not path.exists():
    path.mkdir()
    for o in plant_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} senecio')
        download_images(dest, urls=results.attrgot('contentUrl'))


# In[9]:


fns = get_image_files(path)
fns


# In[10]:


failed = verify_images(fns)
failed


# In[11]:


failed.map(Path.unlink);


# In[12]:


plants = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))


# In[13]:


dls = plants.dataloaders(path)


# In[14]:


dls.valid.show_batch(max_n=4, nrows=1)


# In[15]:


plants = plants.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = plants.dataloaders(path)


# In[16]:


plants = plants.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = plants.dataloaders(path)


# In[17]:


learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)


# In[18]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[20]:


cleaner = ImageClassifierCleaner(learn)
cleaner


# In[19]:


interp.plot_top_losses(5, nrows=1)


# In[21]:


learn.export()


# In[22]:


path = Path()
path.ls(file_exts='.pkl')


# In[23]:


learn_inf = load_learner(path/'export.pkl')


# In[24]:


#hide_output
btn_upload = widgets.FileUpload()
btn_upload


# In[25]:


img = PILImage.create(btn_upload.data[-1])


# In[26]:


#hide_output
out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl


# In[27]:


pred,pred_idx,probs = learn_inf.predict(img)


# In[28]:


#hide_output
lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred


# In[30]:


#hide
get_ipython().system('pip install voila')
#jupyter serverextension enable --sys-prefix voila 


# In[31]:


get_ipython().system('jupyter serverextension enable --sys-prefix voila ')


# In[ ]:




