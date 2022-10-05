#!/usr/bin/env python3

import sys
num_token = sys.argv[1]

string = ' '.join(list(map(str, list(range(1, int(num_token)+1)))))

print('''<Topology>
<TopologyEntry>
<ForPhones>
%s
</ForPhones>
<State> 0 <PdfClass> 0 <Transition> 1 1.0 </State>
<State> 1 </State>
</TopologyEntry>
</Topology>
''' % (string))
