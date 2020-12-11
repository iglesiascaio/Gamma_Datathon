arc_rename_dict = { 
    'Identifiant arc' : 'arc_id',
    'Libelle' : 'tag', 
    'Débit horaire' : 'tr_flow',
    'Taux d\'occupation' : 'occu_rate',
    'Etat trafic' : 'tr_state',
    'Identifiant noeud amont' : 'v_start_id',
    'Libelle noeud amont' : 'v_start_tag',
    'Identifiant noeud aval' : 'v_end_id',
    'Libelle noeud aval' : 'v_end_tag',
    'Etat arc' : 'arc_state',
    'Date debut dispo data' : 'start_available_date',
    'Date fin dispo data' : 'end_available_date',
    'geo_point_2d' : 'geo_point_2d',
    'geo_shape' : 'geo_shape'
}

# Dictionary defining in which periods certain events occured, and the value we associate to them
periods_dict = {
    'lockdown_intensity':{
        ('2020-03-17', '2020-05-11') : 1,   # First lockdown in France 
        ('2020-10-30', '2020-11-27') : 0.4, # Second lockdown in France (less restrictions => less intense)
        ('2020-11-28', '2020-12-15') : 0.2  # Alleviation of second lockdown
    },
    'vacation':{
        # Source: https://www.education.gouv.fr/sites/default/files/imported_files/document/2017-2018-494.pdf
        ('2015-10-18' , '2015-11-01') : 1,  # Toussaints
        ('2015-12-20' , '2016-01-03') : 1,  # Noel
        ('2016-02-21' , '2016-03-06') : 1,  # 
        ('2016-04-17' , '2016-04-30') : 1,
        ('2016-07-06' , '2016-09-02') : 1,  # Été 
        # Source: https://www.education.gouv.fr/sites/default/files/imported_files/document/2017-2018-494.pdf
        ('2016-10-20' , '2016-11-02') : 1,  # Toussaints
        ('2016-12-18' , '2017-01-02') : 1,  # Noel
        ('2017-02-05' , '2017-02-19') : 1,  # 
        ('2017-04-02' , '2017-04-17') : 1,
        ('2017-07-09' , '2017-09-03') : 1,  # Été 
        # Source: https://www.education.gouv.fr/sites/default/files/imported_files/document/2017-2018-494.pdf
        ('2017-10-22' , '2017-11-05') : 1,  # Toussaints
        ('2017-12-24' , '2018-01-07') : 1,  # Noel
        ('2018-02-18' , '2018-03-04') : 1,  # 
        ('2018-04-15' , '2018-04-29') : 1,
        ('2018-07-08' , '2018-09-02') : 1,  # Été 
        # Source: https://www.education.gouv.fr/sites/default/files/imported_files/document/2018-2019-493.pdf
        ('2018-10-21' , '2018-11-04') : 1,  # Toussaints
        ('2018-12-23' , '2019-01-06') : 1,  # Noel
        ('2019-02-24' , '2019-03-10') : 1,  # 
        ('2019-04-21' , '2019-05-05') : 1,
        ('2019-05-30' , '2019-06-02') : 1,  #
        ('2019-07-07' , '2019-08-31') : 1,  # Été 
        # Source: https://www.education.gouv.fr/sites/default/files/2020-02/calendrier-scolaire-2019-2020-pdf-44402.pdf
        ('2019-10-20' , '2019-11-03') : 1,  # Toussaints
        ('2019-12-22' , '2020-01-05') : 1,  # Noel
        ('2020-02-09' , '2020-02-23') : 1,  # 
        ('2020-04-05' , '2020-04-19') : 1,
        ('2020-05-21' , '2020-05-24') : 1,  #
        ('2020-07-05' , '2020-08-31') : 1,  # Été 
        # Source: holidays python library
        ('2019-01-01' , '2019-01-01') : 1,  #Jour de l'an
        ('2019-05-01' , '2019-05-01') : 1,  #Fête du Travail
        ('2019-05-08' , '2019-05-08') : 1,  #Armistice 1945
        ('2019-07-14' , '2019-07-14') : 1,  #Fête nationale
        ('2019-11-11' , '2019-11-11') : 1,  #Armistice 1918
        ('2019-04-22' , '2019-04-22') : 1,  #Lundi de Pâques
        ('2019-06-10' , '2019-06-10') : 1,  #Lundi de Pentecôte
        ('2019-05-30' , '2019-05-30') : 1,  #Ascension
        ('2019-08-15' , '2019-08-15') : 1,  #Assomption
        ('2019-11-01' , '2019-11-01') : 1,  #Toussaint
        ('2019-12-25' , '2019-12-25') : 1,  #Noël
        ('2020-01-01' , '2020-01-01') : 1,  #Jour de l'an
        ('2020-05-01' , '2020-05-01') : 1,  #Fête du Travail
        ('2020-05-08' , '2020-05-08') : 1,  #Armistice 1945
        ('2020-07-14' , '2020-07-14') : 1,  #Fête nationale
        ('2020-11-11' , '2020-11-11') : 1,  #Armistice 1918
        ('2020-04-13' , '2020-04-13') : 1,  #Lundi de Pâques
        ('2020-06-01' , '2020-06-01') : 1,  #Lundi de Pentecôte
        ('2020-05-21' , '2020-05-21') : 1,  #Ascension
        ('2020-08-15' , '2020-08-15') : 1,  #Assomption
        ('2020-11-01' , '2020-11-01') : 1,  #Toussaint
        ('2020-12-25' , '2020-12-25') : 1,   #Noël
    }
}

