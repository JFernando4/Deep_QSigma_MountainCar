
def check_attribute_else_default(object_type, attr_name, default_value):
    if not hasattr(object_type, attr_name):
        print("Creating attribute", attr_name)
        setattr(object_type, attr_name, default_value)
    return getattr(object_type, attr_name)


def check_dict_else_default(dict_type, key_name, default_value):
    assert isinstance(dict_type, dict)
    if key_name not in dict_type.keys():
        dict_type[key_name] = default_value
    return dict_type[key_name]
