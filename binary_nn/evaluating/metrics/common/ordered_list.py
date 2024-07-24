

def ordered_list(values, name_template):
    return {name_template.format(i=i, val=val): val for i, val in enumerate(values)}
