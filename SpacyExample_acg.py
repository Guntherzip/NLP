# AIA SpaCy Use.py
# @copyright: Collin Lynch
#
# SpaCY is a pipeline-based NLP module for python.  In contrast to the NLTK
# or Gensim which are tool oriented SpaCY is built around the idea of loading
# in trained language models and then using them for all of your tasks.  In
# this way it makes SpaCy relatively easier to use inside of a system,
# particularly if you are interested in off-the-shelf solutions.  But it also
# means that the details of what is happening and how it is happening are
# obscured.
#
# It can also be overkill as the spacy pipeline approach does a *lot* of
# processing at one time which may be much more than you need if you are
# only interested in tokenization, vectors, etc.  All of those items can
# be built into your own model for use but they will vary.


# Installation of Spacy requires both the module and downloading of the pretrained
# items which must be built separately and then loaded into the system.  Spacy
# has many different language models built in and can be used for a range of tasks
# including tokenizing POS etc. as we have shown.

# To install spacy use anaconda or other tasks to load it;
# pip install --user spacy

# Then to install the libraries you have to download it separately.  To do that
# go to the anaconda prompt and make the following call.  
#
#  python -m spacy download en_core_web_sm

# If you get an error about needing to call pip directly you can do
# the following at the commandline:
#
#  pip install --user https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0-py3-none-any.whl
#
# Optionally you can try to download it inside of python using the
# following however that too is error prone.
#


# Once inside the system you can go ahead and import Spacy then load
# the pretrained language model.  
import spacy
spacy.cli.download("en_core_web_sm")



# Having installed the model we now load it as an object.  In this case
# the type of the model is spacy.lang.en.English which is a subclass of
# the general spacy language model.  You can learn more about it as always
# via the help function. 
# https://spacy.io/models/en
Model = spacy.load("en_core_web_sm")


# Here we will illustrate the model on some basic text from the
# existing NLTK corpus.  In this case we get the raw text and
# then we will process that.
import nltk
nltk.download('gutenberg')
WhitmanRaw = nltk.corpus.gutenberg.raw(fileids="whitman-leaves.txt")
#print(WhitmanRaw)

# Once we generate the model and load some raw text we will use the
# model to parse our text into a doc object.  This object is the
# main item around which spacy is built.  The doc object does basic
# parsing using the pretrained model and gives us access to different
# pieces of output.  To generate a doc we simply call the Model as
# a function as it is a generator.
#
# [This can be quite slow.]
#
# The resulting item's type will be: spacy.tokens.doc.Doc which
# encapsulates a lot of what we want to achieve.  
WhitmanDoc = Model(WhitmanRaw)
print(type(WhitmanDoc))


# Once the Doc is created and the parsing done you can carry out some
# basic extraction tasks using the standard interface.  Under the hood
# the doc is actually represented by vector/tensor objects but it also
# holds all of the tokens and other items so we can extract tokens or
# subsequences from it using standard Python notation:

Word = WhitmanDoc[1023]
print(Word)

Subset = WhitmanDoc[23:56]
print(Subset)


# The items returned are not just simple strings, they are in fact
# spacy.tokens.token.Token objects which store information about
# them including the POS tags, vector position etc. 
print('Type = ', type(Word))

print(Word.text)

# Here is the part of speech.  Note the version without the _ is
# an integer id of the unique POS in the model while the _ version
# is the human-readable string.
print(Word.pos)
print(Word.pos_)

# Print whether this is in the predefined list of stopwords.
print(Word.is_stop)

# And print out the vector embedding.  
if Word.has_vector:
    print(Word.vector)


# Having precomputed a model we can also save the file on disk.
#WhitmanDoc.to_disk("./Save/Whitman")
WhitmanDoc.to_disk("whitman")


# To load it we must preload a vocab object to import it.  Since
# we used the existing defaults that is straightforward.
BaseVocab = spacy.vocab.Vocab()
LoadedDoc = spacy.tokens.Doc(BaseVocab).from_disk("whitman")



# Embedded in the doc parse is a set of the sentence dependencies
# which we can use to get at some of the grammar information.  We
# can show what this looks like on a smaller example using the
# displacy library which is built into spacy.

from spacy import displacy

DepDoc = Model("This is difficult to write but, I ate your doughnut.")


# If that is not viewable you can also save the svg file which is an XML
# representation for vector graphics that is viewable by browsers.  So once
# you call this code to save the file you will need to open it in a browser.

File = displacy.render(DepDoc, style='dep')
Out = open("ImageSave.svg", "w", encoding="utf-8")
Out.write(File)
Out.close()


# Once run this will open a local web server that you can use to view it.
# alternative display methods can also be played with.
# this will display on your base IP address on port 5000
# open a new browser window and enter the IP address:5000
# press stop when finished
displacy.serve(DepDoc, style='dep')
