=====
Usage
=====

The easiest way to use :py:mod:`orsopy.fileio` (the module of :py:mod:`orsopy` that includes file reading and writing) to produce **metadata-rich .ort reduced reflectometry files** involves integrating this into your data reduction workflow.
Early in the workflow, the :py:mod:`orsopy.fileio` should be imported and an empty :code:`orsopy.fileio.orso.Orso` header object (here we also import :py:mod:`numpy` which will be used later).

.. code-block:: python

    from orsopy import fileio
    from orsopy.fileio import (Reduction, Person, Orso, Experiment,
                               Sample, DataSource, Measurement, InstrumentSettings)
    import numpy as np
    import datetime

    header = fileio.orso.Orso.empty()

Having created the empty header object we can start to populate the appropriate components of it.
It is generally a good idea to populate the components as particular steps occur in the reduction process.
In this example we fill out the experiment details and then wrap them in the appropriate class.

.. code-block:: python

    title = 'my title'
    instrument = 'PLATYPUS'
    start_date = datetime.date.today()  # This needs to be an object of the type datetime
    probe = 'neutron'
    facility = 'ESS'
    proposalID = 999999
    doi = f'10.5286/ISIS.E.RB{99999}'

    experiment = Experiment(title, instrument, start_date, probe, facility, proposalID, doi)

Full details of the different components that can be populated can be found in the `documentation`_ here or in the `file format specification`_.
Note that this specification includes information regarding the **required** and optional components to be included for a file to be considered a **valid** .ort file.
What we show here is not the minimum needed to write an `.ort` file, rather the minium to fill out all the major fields with information which is likely to be available at time of writing.

We now write out the user and sample details, but note that the sample can be alot more descriptive than just a name.

.. code-block:: python

    user_name = 'Jos Cooper'
    user_affil = 'ESS'

    user = Person(user_name, user_affil)

    # Now fill the sample details
    sample_name = 'Silicon'

    sample = Sample(sample_name)

    # The measurement details needs some instrument settings assigned first
    angle = 0.3  # This can be a float, or a range, though currently not a list
    wavelength = (1.8, 8.8)  # Again a float or range

    instrument_settings = InstrumentSettings(angle, wavelength)

    # Now add the instrument settings to the names of the files used
    files = ['test_data.raw']  # A list of all raw files used reduced

    measurment = Measurement(instrument_settings, files)

We can now wrap together the user, experiment, sample and measurement information into a single object.

.. code-block:: python

    data_source_info = DataSource(user, experiment, sample, measurment)

It is not possible to write an `.ort` file without defining the columns present in the dataset, in this example we will have four columns of data, namely q, R, dR and dq (the final column is a description of the resolution function).
Columns are defined as follows, using the :code:`orsopy.fileio.base.Column` and :code:`orsopy.fileio.base.ErrorColumn` class objects.

.. code-block:: python

    # Interpreted units are ["1/angstrom", "1/nm", "1", "1/s", None]
    q_column = fileio.base.Column(name='Qz', unit='1/angstrom', physical_quantity='wavevector transfer')
    r_column = fileio.base.Column(name='R', unit=None, physical_quantity='reflectivity')
    dr_column = fileio.base.ErrorColumn(error_of='R', error_type='uncertainty', value_is='sigma')
    dq_column = fileio.base.ErrorColumn(error_of='Qz', error_type='resolution', value_is='sigma')

    header.columns = [q_column, r_column, dr_column, dq_column]
    # We can also make some data so that this code example will write something out
    q = np.array([0.01,0.02,0.03])
    R = np.array([0.1,0.2,0.3])
    dR = np.array([0.001,0.002,0.003])
    dq = q * 0.02

Any **required** metadata that is not included in the head will be written in the file as containing :code:`null`.


Now, we then want to assign the data that we want to write (this will be after your data reduction has been performed).
This is achieved by producing a :code:`fileio.orso.OrsoDataset` object, which takes the header and the relevant data columns (below these are :code:`q`, :code:`R`, :code:`dR`, and :code:`dq`) as inputs.

.. code-block:: python

    orso_class = Orso(data_source_info, reduction=Reduction('My own code'), columns=header.columns)  # reduction can also be assigned out of this funciton call
    dataset = fileio.orso.OrsoDataset(info=orso_class, data=np.array([q, R, dR, dq]).T)

The dataset can then be saved with the following function, where :code:`'my_file.ort'` is the name for the file to be saved under.

.. code-block:: python

    fileio.orso.save_orso(datasets=[dataset], fname='my_file.ort')  # note that the first input is a list of datasets

Note that if you want to save more than one dataset in a single file, this can be achieved by including these in the list that is passed to this function.


.. _`documentation`: ./modules.html#fileio
.. _`file format specification`: https://www.reflectometry.org/file_format/specification
.. _`base classes`: ./orsopy.fileio.base.html
