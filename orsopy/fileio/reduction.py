"""
The reduction elements for the ORSO header
"""

import datetime

from dataclasses import dataclass, field
from typing import List, Optional, Union

from .base import Header, Person


@dataclass(repr=False)
class Software(Header):
    """
    Software description.

    :param name: Software name.
    :param version: Version identified for the software.
    :param platform: Operating system.
    """

    name: str
    version: Optional[str] = None
    platform: Optional[str] = None

    yaml_representer = Header.yaml_representer_compact


@dataclass(repr=False)
class Reduction(Header):
    """
    A description of the reduction that has been performed.

    :param software: Software used for reduction.
    :param timestamp: Datetime of reduced file creation.
    :param creator: The person or routine who created the reduced file.
    :param corrections: A list of the corrections that have been performed.
    :param computer: Name of the reduction machine.
    :param call: Command line call or similar.
    :param script: Path to reduction script or notebook.
    :param binary: Path to full reduction information file.
    """

    software: Software
    timestamp: Optional[datetime.datetime] = field(
        default=None, metadata={"description": "Timestamp string, formatted as ISO 8601 datetime"}
    )
    creator: Optional[Person] = None
    corrections: Optional[List[str]] = None
    computer: Optional[str] = field(default=None, metadata={"description": "Computer used for reduction"})
    call: Optional[str] = field(default=None, metadata={"description": "The command line call used"})
    script: Optional[str] = field(default=None, metadata={"description": "Path to reduction script or notebook"})
    binary: Optional[str] = field(default=None, metadata={"description": "Path to full information file"})

    __repr__ = Header._staggered_repr
