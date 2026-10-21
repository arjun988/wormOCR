"""A small built-in English word list for synthetic lines.

This is not a language model. It only gives the renderer realistic letter
sequences. Random strings still cover the rest of the charset.
"""

from __future__ import annotations

WORDS = """
a about above after again against all almost along already also always am
an and another any are around as at back be because been before being below
between both but by came can come could day did do does doing done down during
each early even ever every few find first for from further get give go good
great had has have he her here him his how i if in into is it its just keep
know large last later least left let like little long look made make many may
me might more most much must my near need never new next no not now of off
often old on once one only or other our out over own part people place put
right said same say see she should since small so some still such take than
that the their them then there these they this those through time to today
too under until up upon us use very want was water way we well were what when
where which while who why will with within without work would year you your
able across add age ago air allow alone along already amount answer appear
area arm ask away bad ball bank base become began begin behind believe best
better big bill black blue body book both box bring build business call car
care carry case catch cause center certain change check child city class clear
close color come common company complete consider continue control cost could
country course cover create cross current dark data day decide deep describe
design detail develop difference difficult direct door down draw drive drop
during each early earth east easy eat edge effect effort eight either else end
energy enough enter equal error especially evening ever every example expect
experience explain eye face fact fail fall family far fast father feel few
field figure fill final find fine finger finish fire firm first fish five
floor follow food foot force form forward found four free friend from front
full fund further future game garden general get girl give glass go goal
gold gone good government great green ground group grow guess half hand happen
happy hard have head health hear heart heat heavy help her high history hold
home hope horse hour house however human hundred idea important include
indeed information interest into issue item itself job join just keep key
kind king kitchen know land language large last late later laugh law lay lead
learn leave left leg less letter level lie life light like likely line list
listen little live local long look lose loss lot love low machine main major
make man many map mark market material matter may mean measure meet member
men mention middle might mile mind minute miss model modern moment money
month more morning most mother move much music must my name nation nature
near nearly need never new news next night nine no none nor north not note
nothing notice now number of off offer office often oil old on once one only
open or order other our out outside over own page paint paper parent part
party pass past pay people per perhaps period person phone pick picture
piece place plan plant play please point police policy poor position possible
power practice prepare present president pretty price private probably problem
process produce product program project provide public pull push put quality
question quick quite race radio raise range rate rather reach read ready real
really reason receive recent record red remain remember remove report require
research rest result return rich right rise road rock room rule run safe said
same save say school science sea season seat second see seem sell send sense
sentence serve service set seven several shake shall share she short should
show side sign simple since sing single sister sit six size skill small smile
so social society some son song soon sort sound south space speak special
spend stand start state stay step still stop story street strong student study
such sudden suggest summer sun support sure system table take talk tax teach
team tell ten term test than thank that the their them then theory there these
they thing think third this those though thought three through throughout throw
thus time to today together too took top total toward town trade train travel
tree trial trip trouble true try turn two type under understand unit until up
upon us use usually value various very view visit voice wait walk wall want
war was watch water way we week well west what whatever when where whether
which while white who whole why wide wife will win wind window wish with
within without woman wonder word work world would write wrong year yes yet you
young your yourself
bookkeeper committee mississippi balloon coffee letter success address moon
feed cool tall egg too see book keep good add fee all fall call hill bell
yellow apple queen zebra jazz fox box quick jumps lazy dog vex wax oxygen
""".split()

REPEAT_WORDS = [
    "bookkeeper",
    "committee",
    "mississippi",
    "balloon",
    "coffee",
    "letter",
    "success",
    "address",
    "moon",
    "feed",
    "cool",
    "tall",
    "egg",
    "see",
    "too",
    "good",
    "add",
    "all",
    "book",
    "keep",
    "fee",
    "hill",
    "bell",
    "yellow",
    "apple",
    "jazz",
    "queen",
]

PANGRAM = "The quick brown fox jumps over the lazy dog"
