import subprocess

def generate_airprint_printer():
    # Get the list of shared printers from CUPS
    printers = subprocess.check_output(['lpstat', '-v']).decode('utf-8').split('\n')
    print(f'Printers: {printers}')
    shared_printers = printers[0].split()

    #shared_printers = [p.split()[0] for p in printers if 'shared' in p]
    print(f'Shared printers: {shared_printers}')

    # Generate Avahi service files for each shared printer
    for printer in shared_printers:
        print(f'Printer: {printer}')
        service_file = f'/etc/avahi/services/airprint-{printer}.service'
        with open(service_file, 'w+') as f:
            f.write(f'''<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">AirPrint {printer}</name>
  <service>
    <type>_ipp._tcp</type>
    <port>631</port>
    <txt-record>txtvers=1</txt-record>
    <txt-record>qtotal=1</txt-record>
    <txt-record>rp=printers/{printer}</txt-record>
    <txt-record>ty={printer}</txt-record>
    <txt-record>adminurl=http://localhost:631/printers/{printer}</txt-record>
    <txt-record>note=Print from your iPhone or iPad</txt-record>
  </service>
</service-group>
''')

if __name__ == '__main__':
    generate_airprint_printer()
