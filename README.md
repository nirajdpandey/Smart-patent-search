# Smart-patent-search

Search for a right patent provided any information in any order via terminal

### Purpose
Patent documents contain lots of different types of information and standard search
functionalities have search fields that are limited to one information type. This makes it
hard for less experienced patent researches to find the right patent.

### Goal

Develop a prototype which can support following requirements 

1. That searches the different types of information that patent documents contain
	without having the user use field identifiers. The most used information types that
	should be included are: Title, Abstract, Description, Claims, Inventors/Applicants,
	IPC number, CPC number, Priority number, Publication date

2. That offers a semantic search that searches not only for exact keywords but the
	meaning and similarity in semantic content.

### Data 

The glimpse of data can be seen in the folder called data. However, that;s not complete corpus. 
If you wish to download complete set here is the [link](https://publication.epo.org/raw-
data/download/files/2019/03/27/1553702348281/EP_full_text_backfile_sample_data_1980.zip). 

### Approach

```
1. Extract text from the native .xml patent files
2. Remove useless information
3. Get pre-trained Word Embedding
4. Search for similar document given some search term
5. Find least distances using Word Moving Distances (WMD)
6. Sort distances in Decreasing order
7. Find The index where distance was least
8. Return that patent file

```
