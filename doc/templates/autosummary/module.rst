{{ objname | escape | underline(line="=") }}

.. automodule:: {{ fullname }}

    {% block classes %}
    {% if classes %}
    .. rubric:: {{ _('Classes') }}

    .. autosummary::
        :toctree:
    {% for item in classes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}


    {% block functions %}
    {% if functions %}
    .. rubric:: {{ _('Functions') }}

    .. autosummary::
        :toctree:
    {% for item in functions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}
