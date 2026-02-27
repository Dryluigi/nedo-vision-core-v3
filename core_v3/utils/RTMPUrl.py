class RTMPUrl:
    """
    Call `configure()` static method first.
    """
    _rtmp_server_url = ""
    _rtmp_publish_query_strings = ""

    @staticmethod
    def configure(rtmp_server_url: str, rtmp_publish_query_strings: str = ""):
        """
        Example:
        - `rtmp_server_url` = rtmp://live.sindika.co.id:8554/live
        - `rtmp_publish_query_strings` = username=my-username&pass=mantap-jiwa
        """
        RTMPUrl._rtmp_server_url = rtmp_server_url
        RTMPUrl._rtmp_publish_query_strings = rtmp_publish_query_strings
        
    @staticmethod
    def get_publish_url(stream_key: str) -> str:
        url = f"{RTMPUrl._rtmp_server_url}/{stream_key}"
        if RTMPUrl._rtmp_publish_query_strings:
            url = f"{RTMPUrl._rtmp_server_url}/{stream_key}?{RTMPUrl._rtmp_publish_query_strings}"

        return url