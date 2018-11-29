import dropbox
import os
import sys

def upload(path) :
    dbx = dropbox.Dropbox("YSNAFwjBvHAAAAAAAAAAzLC75UxggJTFqXuGDI-W71k_-FfDd-yeA2GS8weuy61C")
    # print('linked account: ', dbx.users_get_current_account())

    file_path = path
    file_name = file_path.split('/')[-1]
    dest_path = '/'+file_name

    f = open(file_path,'rb')
    file_size = os.path.getsize(file_path)

    CHUNK_SIZE = 4 * 1024 * 1024

    if file_size <= CHUNK_SIZE:

        dbx.files_upload(f.read(), dest_path)
        print("UPLOADED")

    else:

        upload_session_start_result = dbx.files_upload_session_start(f.read(CHUNK_SIZE))

        cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                   offset=f.tell())
        commit = dropbox.files.CommitInfo(path=dest_path)

        while f.tell() < file_size:
            print("\r Uploading - "+str(f.tell())+"/"+str(file_size)+" : "+str(int(f.tell()/file_size*100))+"%",end="")
            if ((file_size - f.tell()) <= CHUNK_SIZE):
                print(dbx.files_upload_session_finish(f.read(CHUNK_SIZE),
                                                cursor,
                                                commit))
                print("UPLOADED")
            else:
                dbx.files_upload_session_append_v2(f.read(CHUNK_SIZE),
                                                cursor)
                cursor.offset = f.tell()

if __name__ == '__main__':
    path = sys.argv
    if (len(path)==1) :
        print("File not found")
    else :
        upload(path[-1])