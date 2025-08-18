# CyVerse iRODS

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
CyVerse storage is accessed through iRODS paths handled by the `gocmd` utility. Remote locations are written with an `i:` prefix followed by the iRODS zone, for example `i:/iplant/home/your_username`.

To authenticate, run `./gocmd init` once. The command records your credentials in `~/.irods` so future operations can reach the data store without re-entering them.

Common operations:

```bash
# list a collection
./gocmd ls i:/iplant/home/your_username

# download a file
./gocmd get i:/iplant/home/your_username/data.txt

# upload a file to a collection
./gocmd put local_file.txt i:/iplant/home/your_username/
```

Scripts build remote paths using variables patterned as:

```python
remote_path = f"i:/iplant/{remote_prefix}/{dest_path}"
```

Replace `remote_prefix` and `dest_path` with the appropriate subdirectory and filename for your project.
<!-- FILLME:END -->
