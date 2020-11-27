#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import datajoint as dj

schema = dj.schema('nnfabrik_katharina_tutorial', locals())


@schema
class Monkey(dj.Manual):
      definition = """
      # monkey
      monkey_id: int                  # unique mouse id
      ---
      sex: enum('M', 'F', 'U')    # sex of monkey - Male, Female, or Unknown/Unclassified
      age: int                      #age of monkey
      """

